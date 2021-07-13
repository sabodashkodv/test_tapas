import os
import shutil
import time
from tapas.utils import tf_example_utils
from tapas.run_task_main import TpuOptions

from tapas_component.component import make_prediction, model_preparation
from tapas_component.utils import convert_interactions_to_examples, write_tf_example, show_answer

model_root = 'tapas_sqa_small'

os.makedirs('results/sqa/tf_examples', exist_ok=True)
os.makedirs('results/sqa/model', exist_ok=True)

with open('results/sqa/model/checkpoint', 'w') as f:
    f.write('model_checkpoint_path: "model.ckpt-0"')
for suffix in ['.data-00000-of-00001', '.index', '.meta']:
    shutil.copyfile(f'{model_root}/model.ckpt{suffix}', f'results/sqa/model/model.ckpt-0{suffix}')

max_seq_length = 512
vocab_file = f"{model_root}/vocab.txt"
config = tf_example_utils.ClassifierConversionConfig(
    vocab_file=vocab_file,
    max_seq_length=max_seq_length,
    max_column_id=max_seq_length,
    max_row_id=max_seq_length,
    strip_column_names=False,
    add_aggregation_candidates=False,
)
converter = tf_example_utils.ToClassifierTensorflowExample(config)


def run_question_answer(table_data, queries, task, converter):
    table_data = table_data.astype(str)
    table = [[]]
    table[0] = list(table_data.columns)
    table.extend(table_data.values.tolist())
    examples = convert_interactions_to_examples([(table, queries)], converter)
    write_tf_example("results/sqa/tf_examples/test.tfrecord", examples)
    write_tf_example("results/sqa/tf_examples/random-split-1-dev.tfrecord", [])

    output_dir = os.path.join('results', task.name.lower())
    model_dir = os.path.join(output_dir, 'model')

    tpu_options = TpuOptions(
        use_tpu=False,
        tpu_name=None,
        tpu_zone=None,
        gcp_project=None,
        master=None,
        num_tpu_cores=8,
        iterations_per_loop=1000)
    start_model_preparation = time.time()
    estimator, task, model_dir, tapas_config, do_model_classification, do_model_aggregation, \
    use_answer_as_supervision = model_preparation(task,
                                                  tpu_options,
                                                  test_batch_size=len(queries),
                                                  train_batch_size=None,
                                                  gradient_accumulation_steps=1,
                                                  bert_config_file=f"{model_root}/bert_config.json",
                                                  init_checkpoint=f"{model_root}/model.ckpt",
                                                  test_mode=False,
                                                  model_dir=model_dir)
    print(f'{model_root} model_preparation take {time.time() - start_model_preparation} seconds')
    start_model_predict = time.time()
    make_prediction(estimator, task, False, model_dir, output_dir, tapas_config, do_model_classification,
                    do_model_aggregation, use_answer_as_supervision, do_eval=False)
    print(f'{model_root} model_prediction take {time.time() - start_model_predict} seconds')

    return show_answer(table, queries)
