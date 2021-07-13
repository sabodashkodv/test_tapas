import os
import shutil

import time
from tapas.utils import tf_example_utils, tasks
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

task = tasks.Task.SQA
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


def predict(table_data, queries, task, converter):
    table = [list(map(lambda s: s.strip(), row.split("|")))
             for row in table_data.split("\n") if row.strip()]
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


result = predict("""
Pos | No | Driver               | Team                           | Laps | Time/Retired | Grid | Points
1   | 32 | Patrick Carpentier   | Team Player's                  | 87   | 1:48:11.023  | 1    | 22    
2   | 1  | Bruno Junqueira      | Newman/Haas Racing             | 87   | +0.8 secs    | 2    | 17    
3   | 3  | Paul Tracy           | Team Player's                  | 87   | +28.6 secs   | 3    | 14
4   | 9  | Michel Jourdain, Jr. | Team Rahal                     | 87   | +40.8 secs   | 13   | 12
5   | 34 | Mario Haberfeld      | Mi-Jack Conquest Racing        | 87   | +42.1 secs   | 6    | 10
6   | 20 | Oriol Servia         | Patrick Racing                 | 87   | +1:00.2      | 10   | 8 
7   | 51 | Adrian Fernandez     | Fernandez Racing               | 87   | +1:01.4      | 5    | 6
8   | 12 | Jimmy Vasser         | American Spirit Team Johansson | 87   | +1:01.8      | 8    | 5
9   | 7  | Tiago Monteiro       | Fittipaldi-Dingman Racing      | 86   | + 1 Lap      | 15   | 4
10  | 55 | Mario Dominguez      | Herdez Competition             | 86   | + 1 Lap      | 11   | 3
11  | 27 | Bryan Herta          | PK Racing                      | 86   | + 1 Lap      | 12   | 2
12  | 31 | Ryan Hunter-Reay     | American Spirit Team Johansson | 86   | + 1 Lap      | 17   | 1
13  | 19 | Joel Camathias       | Dale Coyne Racing              | 85   | + 2 Laps     | 18   | 0
14  | 33 | Alex Tagliani        | Rocketsports Racing            | 85   | + 2 Laps     | 14   | 0
15  | 4  | Roberto Moreno       | Herdez Competition             | 85   | + 2 Laps     | 9    | 0
16  | 11 | Geoff Boss           | Dale Coyne Racing              | 83   | Mechanical   | 19   | 0
17  | 2  | Sebastien Bourdais   | Newman/Haas Racing             | 77   | Mechanical   | 4    | 0
18  | 15 | Darren Manning       | Walker Racing                  | 12   | Mechanical   | 7    | 0
19  | 5  | Rodolfo Lavin        | Walker Racing                  | 10   | Mechanical   | 16   | 0
""", ["what were the drivers names?",
      "of these, which points did patrick carpentier and bruno junqueira score?",
      "who scored higher?"], task, converter)
