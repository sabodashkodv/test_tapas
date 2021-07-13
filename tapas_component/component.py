import enum
import functools
import os
import random
import time

import tensorflow.compat.v1 as tf
from tapas.experiments import prediction_utils as exp_prediction_utils
from tapas.models import tapas_classifier_model
from tapas.models.bert import modeling
from tapas.scripts import calc_metrics_utils
from tapas.scripts import prediction_utils
from tapas.utils import file_utils
from tapas.utils import hparam_utils
from tapas.utils import number_annotation_utils
from tapas.utils import pruning_utils
from tapas.utils import task_utils
from tapas.utils import tasks
from tapas.utils import tf_example_utils

from config import FLAGS
from tapas_component.utils import _print, _warn
from tapas_component.enums import Task, Mode, TestSet

tf.disable_v2_behavior()

_MAX_TABLE_ID = 512
_MAX_PREDICTIONS_PER_SEQ = 20
_CELL_CLASSIFICATION_THRESHOLD = 0.5


def model_preparation(task,
                      tpu_options,
                      test_batch_size,
                      train_batch_size,
                      gradient_accumulation_steps,
                      bert_config_file,
                      init_checkpoint,
                      test_mode,
                      model_dir, ):
    file_utils.make_directories(model_dir)
    num_aggregation_labels, num_classification_labels, use_answer_as_supervision = set_task_params(task)

    do_model_aggregation = num_aggregation_labels > 0
    do_model_classification = num_classification_labels > 0

    hparams = hparam_utils.get_hparams(task)
    test_batch_size, train_batch_size, num_train_steps, num_warmup_steps = get_train_params(test_mode, train_batch_size,
                                                                                            hparams, test_batch_size)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tapas_config = tapas_classifier_model.TapasClassifierConfig(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=hparams['learning_rate'],
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=tpu_options.use_tpu,
        positive_weight=10.0,
        num_aggregation_labels=num_aggregation_labels,
        num_classification_labels=num_classification_labels,
        aggregation_loss_importance=1.0,
        use_answer_as_supervision=use_answer_as_supervision,
        answer_loss_importance=1.0,
        use_normalized_answer_loss=False,
        huber_loss_delta=hparams.get('huber_loss_delta'),
        temperature=hparams.get('temperature', 1.0),
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=tapas_classifier_model.AverageApproximationFunction.RATIO,
        cell_select_pref=hparams.get('cell_select_pref'),
        answer_loss_cutoff=hparams.get('answer_loss_cutoff'),
        grad_clipping=hparams.get('grad_clipping'),
        disabled_features=[],
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=False,
        init_cell_selection_weights_to_zero=hparams['init_cell_selection_weights_to_zero'],
        select_one_column=hparams['select_one_column'],
        allow_empty_column_selection=hparams['allow_empty_column_selection'],
        disable_position_embeddings=False,
        reset_position_index_per_cell=False)

    model_fn = tapas_classifier_model.model_fn_builder(tapas_config)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2

    tpu_cluster_resolver = None
    if tpu_options.use_tpu and tpu_options.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_options.tpu_name,
            zone=tpu_options.tpu_zone,
            project=tpu_options.gcp_project,
        )

    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=tpu_options.master,
        model_dir=model_dir,
        tf_random_seed=None,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=4.0,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=tpu_options.iterations_per_loop,
            num_shards=tpu_options.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # If TPU is not available, this will fall back to normal Estimator on CPU/GPU.
    estimator = tf.estimator.tpu.TPUEstimator(
        params={'gradient_accumulation_steps': gradient_accumulation_steps},
        use_tpu=tpu_options.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size // gradient_accumulation_steps,
        eval_batch_size=None,
        predict_batch_size=test_batch_size)
    return estimator, task, model_dir, tapas_config, do_model_classification, do_model_aggregation, \
           use_answer_as_supervision


def get_train_params(test_mode: bool, train_batch_size: int, hparams: dict, test_batch_size) -> (
        int, int, int, int, int):
    if test_mode:
        if train_batch_size is None:
            train_batch_size = 1
        test_batch_size = 1
        num_train_steps = 10
        num_warmup_steps = 1
    else:
        if train_batch_size is None:
            train_batch_size = hparams['train_batch_size']
        num_train_examples = hparams['num_train_examples']
        num_train_steps = int(num_train_examples / train_batch_size)
        num_warmup_steps = int(num_train_steps * hparams['warmup_ratio'])

    return test_batch_size, train_batch_size, num_train_steps, num_warmup_steps


def set_task_params(task: enum.Enum) -> (int, int, bool):
    if task.value == Task.SQA.value:
        num_aggregation_labels = 0
        num_classification_labels = 0
        use_answer_as_supervision = None
    elif task.value in [
        Task.WTQ.value, Task.WIKISQL.value, Task.WIKISQL_SUPERVISED.value
    ]:
        num_aggregation_labels = 4
        num_classification_labels = 0
        use_answer_as_supervision = task.value != Task.WIKISQL_SUPERVISED.value
    elif task.value == Task.TABFACT.value:
        num_classification_labels = 2
        num_aggregation_labels = 0
        use_answer_as_supervision = True
    else:
        raise ValueError(f'Unknown task: {task.name}')
    return num_aggregation_labels, num_classification_labels, use_answer_as_supervision


def _create_all_examples(
        task,
        vocab_file,
        test_mode,
        output_dir,
        test_batch_size,
):
    """Converts interactions to TF examples."""
    interaction_dir = task_utils.get_interaction_dir(output_dir)
    example_dir = os.path.join(output_dir, 'tf_examples')
    file_utils.make_directories(example_dir)

    _create_examples(
        interaction_dir,
        example_dir,
        vocab_file,
        task_utils.get_train_filename(task),
        batch_size=None,
        test_mode=test_mode)
    _create_examples(interaction_dir, example_dir, vocab_file,
                     task_utils.get_dev_filename(task), test_batch_size,
                     test_mode)
    _create_examples(interaction_dir, example_dir, vocab_file,
                     task_utils.get_test_filename(task), test_batch_size,
                     test_mode)


def _to_tf_compression_type(
        compression_type, ):
    if not compression_type:
        return tf.io.TFRecordCompressionType.NONE
    if compression_type == 'GZIP':
        return tf.io.TFRecordCompressionType.GZIP
    if compression_type == 'ZLIB':
        return tf.io.TFRecordCompressionType.ZLIB
    raise ValueError(f'Unknown compression type: {compression_type}')


def _create_examples(
        interaction_dir,
        example_dir,
        vocab_file,
        filename,
        batch_size,
        test_mode,
):
    """Creates TF example for a single dataset."""

    filename = f'{filename}.tfrecord'
    interaction_path = os.path.join(interaction_dir, filename)
    example_path = os.path.join(example_dir, filename)

    config = tf_example_utils.ClassifierConversionConfig(
        vocab_file=vocab_file,
        max_seq_length=FLAGS.max_seq_length,
        max_column_id=_MAX_TABLE_ID,
        max_row_id=_MAX_TABLE_ID,
        strip_column_names=False,
        add_aggregation_candidates=False,
    )
    converter = tf_example_utils.ToClassifierTensorflowExample(config)

    examples = []
    num_questions = 0
    num_conversion_errors = 0
    for interaction in prediction_utils.iterate_interactions(interaction_path):
        number_annotation_utils.add_numeric_values(interaction)
        for i in range(len(interaction.questions)):
            num_questions += 1

            try:
                examples.append(converter.convert(interaction, i))
            except ValueError as e:
                num_conversion_errors += 1
                _print("Can't convert interaction: %s error: %s", interaction.id,
                       e)
        if test_mode and len(examples) >= 100:
            break

    _print(f'Processed: {filename}')
    _print(f'Num questions processed: {num_questions}')
    _print(f'Num examples: {len(examples)}')
    _print(f'Num conversion errors: {num_conversion_errors}')

    if batch_size is None:
        random.shuffle(examples)
    else:
        # Make sure the eval sets are divisible by the test batch size since
        # otherwise examples will be dropped on TPU.
        # These examples will later be ignored when writing the predictions.
        originial_num_examples = len(examples)
        while len(examples) % batch_size != 0:
            examples.append(converter.get_empty_example())
        if originial_num_examples != len(examples):
            _print(f'Padded with {len(examples) - originial_num_examples} examples.')

    with tf.io.TFRecordWriter(
            example_path,
            options=_to_tf_compression_type(FLAGS.compression_type),
    ) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def _get_train_examples_file(task, output_dir):
    return os.path.join(output_dir, 'tf_examples',
                        f'{task_utils.get_train_filename(task)}.tfrecord')


def _get_test_filename(task, test_set):
    if test_set == TestSet.TEST:
        return task_utils.get_test_filename(task)
    if test_set == TestSet.DEV:
        return task_utils.get_dev_filename(task)
    raise ValueError(f'Unknown test set: {test_set}')


def _get_test_examples_file(
        task,
        output_dir,
        test_set,
):
    filename = _get_test_filename(task, test_set)
    return os.path.join(output_dir, 'tf_examples', f'{filename}.tfrecord')


def _get_test_interactions_file(
        task,
        output_dir,
        test_set,
):
    filename = _get_test_filename(task, test_set)
    return os.path.join(output_dir, 'interactions', f'{filename}.tfrecord')


def _get_test_prediction_file(
        task,
        model_dir,
        test_set,
        is_sequence,
        global_step,
):
    """Get prediction filename for different tasks and setups."""
    suffix = '' if global_step is None else f'_{global_step}'
    if is_sequence:
        suffix = f'_sequence{suffix}'
    filename = _get_test_filename(task, test_set)
    return os.path.join(model_dir, f'{filename}{suffix}.tsv')


def _get_token_selector():
    if not FLAGS.prune_columns:
        return None
    return pruning_utils.HeuristicExactMatchTokenSelector(
        FLAGS.bert_vocab_file,
        FLAGS.max_seq_length,
        pruning_utils.SelectionType.COLUMN,
        # Only relevant for SQA where questions come in sequence
        use_previous_answer=True,
        use_previous_questions=True,
    )


def train(estimator, task, model_dir, output_dir, tapas_config, bert_config, do_model_classification,
          do_model_aggregation, use_answer_as_supervision):
    _print('Training')
    bert_config.to_json_file(os.path.join(model_dir, 'bert_config.json'))
    tapas_config.to_json_file(os.path.join(model_dir, 'tapas_config.json'))
    train_input_fn = functools.partial(
        tapas_classifier_model.input_fn,
        name='train',
        file_patterns=_get_train_examples_file(task, output_dir),
        data_format='tfrecord',
        compression_type=FLAGS.compression_type,
        is_training=True,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=_MAX_PREDICTIONS_PER_SEQ,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=use_answer_as_supervision,
        include_id=False,
    )
    estimator.train(
        input_fn=train_input_fn,
        max_steps=tapas_config.num_train_steps,
    )


def make_prediction(estimator, task, loop_predict, model_dir, output_dir, tapas_config, do_model_classification,
                    do_model_aggregation, use_answer_as_supervision, do_eval=False):
    prev_checkpoint = None
    while True:
        checkpoint = estimator.latest_checkpoint()

        if not loop_predict and not checkpoint:
            raise ValueError(f'No checkpoint found at {model_dir}.')

        if loop_predict and checkpoint == prev_checkpoint:
            _print('Sleeping 5 mins before predicting')
            time.sleep(5 * 60)
            continue

        current_step = int(os.path.basename(checkpoint).split('-')[1])
        _predict(
            estimator,
            task,
            output_dir,
            model_dir,
            do_model_aggregation,
            do_model_classification,
            use_answer_as_supervision,
            use_tpu=tapas_config.use_tpu,
            global_step=current_step,
        )
        if do_eval:
            _eval(
                task=task,
                output_dir=output_dir,
                model_dir=model_dir,
                global_step=current_step)
        if not loop_predict or current_step >= tapas_config.num_train_steps:
            _print(f'Evaluation finished after training step {current_step}.')
            break


def _predict(
        estimator,
        task,
        output_dir,
        model_dir,
        do_model_aggregation,
        do_model_classification,
        use_answer_as_supervision,
        use_tpu,
        global_step,
):
    """Writes predictions for dev and test."""
    for test_set in TestSet:
        _predict_for_set(
            estimator,
            do_model_aggregation,
            do_model_classification,
            use_answer_as_supervision,
            example_file=_get_test_examples_file(
                task,
                output_dir,
                test_set,
            ),
            prediction_file=_get_test_prediction_file(
                task,
                model_dir,
                test_set,
                is_sequence=False,
                global_step=global_step,
            ),
            other_prediction_file=_get_test_prediction_file(
                task,
                model_dir,
                test_set,
                is_sequence=False,
                global_step=None,
            ),
        )
    if task == tasks.Task.SQA:
        if use_tpu:
            _warn('Skipping SQA sequence evaluation because eval is running on TPU.')
        else:
            for test_set in TestSet:
                _predict_sequence_for_set(
                    estimator,
                    do_model_aggregation,
                    use_answer_as_supervision,
                    example_file=_get_test_examples_file(task, output_dir, test_set),
                    prediction_file=_get_test_prediction_file(
                        task,
                        model_dir,
                        test_set,
                        is_sequence=True,
                        global_step=global_step,
                    ),
                    other_prediction_file=_get_test_prediction_file(
                        task,
                        model_dir,
                        test_set,
                        is_sequence=True,
                        global_step=None,
                    ),
                )


def _predict_for_set(
        estimator,
        do_model_aggregation,
        do_model_classification,
        use_answer_as_supervision,
        example_file,
        prediction_file,
        other_prediction_file,
):
    """Gets predictions and writes them to TSV file."""
    # TODO also predict for dev.
    predict_input_fn = functools.partial(
        tapas_classifier_model.input_fn,
        name='predict',
        file_patterns=example_file,
        data_format='tfrecord',
        compression_type=None,
        is_training=False,
        max_seq_length=512,
        max_predictions_per_seq=_MAX_PREDICTIONS_PER_SEQ,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=use_answer_as_supervision,
        include_id=False)
    result = estimator.predict(input_fn=predict_input_fn)
    exp_prediction_utils.write_predictions(
        result,
        prediction_file,
        do_model_aggregation=do_model_aggregation,
        do_model_classification=do_model_classification,
        cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD)
    tf.io.gfile.copy(prediction_file, other_prediction_file, overwrite=True)


def _predict_sequence_for_set(
        estimator,
        do_model_aggregation,
        use_answer_as_supervision,
        example_file,
        prediction_file,
        other_prediction_file,
):
    """Runs realistic sequence evaluation for SQA."""
    examples_by_position = exp_prediction_utils.read_classifier_dataset(
        predict_data=example_file,
        data_format='tfrecord',
        compression_type=None,
        max_seq_length=512,
        max_predictions_per_seq=_MAX_PREDICTIONS_PER_SEQ,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=False,
        add_answer=use_answer_as_supervision)
    result = exp_prediction_utils.compute_prediction_sequence(
        estimator=estimator, examples_by_position=examples_by_position)
    exp_prediction_utils.write_predictions(
        result,
        prediction_file,
        do_model_aggregation,
        do_model_classification=False,
        cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD,
    )
    tf.io.gfile.copy(prediction_file, other_prediction_file, overwrite=True)


def _eval(
        task,
        output_dir,
        model_dir,
        global_step=None,
):
    """Evaluate dev and test predictions."""
    for test_set in TestSet:
        _eval_for_set(
            name=test_set.name.lower(),
            task=task,
            interaction_file=_get_test_interactions_file(
                task,
                output_dir,
                test_set,
            ),
            prediction_file=_get_test_prediction_file(
                task,
                model_dir,
                test_set,
                is_sequence=False,
                global_step=None,
            ),
            global_step=global_step,
        )
        if task == tasks.Task.SQA:
            _eval_for_set(
                name=f'{test_set.name.lower()}_seq',
                task=task,
                interaction_file=_get_test_interactions_file(
                    task,
                    output_dir,
                    test_set,
                ),
                prediction_file=_get_test_prediction_file(
                    task,
                    model_dir,
                    test_set,
                    is_sequence=True,
                    global_step=None,
                ),
                global_step=global_step,
            )


def _eval_for_set(
        name,
        task,
        interaction_file,
        prediction_file,
        global_step,
):
    """Computes eval metric from predictions."""
    if not tf.io.gfile.exists(prediction_file):
        _warn(f"Can't evaluate for {name} because {prediction_file} doesn't exist.")
        return
    test_examples = calc_metrics_utils.read_data_examples_from_interactions(
        interaction_file)
    calc_metrics_utils.read_predictions(
        predictions_path=prediction_file,
        examples=test_examples,
    )
    if task in [
        tasks.Task.SQA, tasks.Task.WTQ, tasks.Task.WIKISQL,
        tasks.Task.WIKISQL_SUPERVISED
    ]:
        denotation_accuracy = calc_metrics_utils.calc_denotation_accuracy(
            examples=test_examples,
            denotation_errors_path=None,
            predictions_file_name=None,
        )
        _print(f'{name} denotation accuracy: {denotation_accuracy:0.4f}')
    elif task == tasks.Task.TABFACT:
        accuracy = calc_metrics_utils.calc_classification_accuracy(test_examples)
        _print(f'{name} accuracy: {accuracy:0.4f}')
    else:
        raise ValueError(f'Unknown task: {task.name}')


def _check_options(output_dir, task, mode):
    """Checks against some invalid options so we can fail fast."""

    if mode == Mode.CREATE_DATA:
        return

    if mode == Mode.PREDICT_AND_EVALUATE or mode == Mode.EVALUATE:
        interactions = _get_test_interactions_file(
            task,
            output_dir,
            test_set=TestSet.DEV,
        )
        if not tf.io.gfile.exists(interactions):
            raise ValueError(f'No interactions found: {interactions}')

    tf_examples = _get_test_examples_file(
        task,
        output_dir,
        test_set=TestSet.DEV,
    )
    if not tf.io.gfile.exists(tf_examples):
        raise ValueError(f'No TF examples found: {tf_examples}')

    _print(f'is_built_with_cuda: {tf.test.is_built_with_cuda()}')
    _print(f'is_gpu_available: {tf.test.is_gpu_available()}')
    _print(f'GPUs: {tf.config.experimental.list_physical_devices("GPU")}')
