from typing import Text, Optional
import dataclasses
import pandas as pd
import csv
import tensorflow.compat.v1 as tf
from absl import logging
from tapas.protos import interaction_pb2
from tapas.scripts import prediction_utils
from tapas.utils import number_annotation_utils


def show_answer(table, queries):
    results_path = "results/sqa/model/test_sequence.tsv"
    all_coordinates = list()
    df = pd.DataFrame(table[1:], columns=table[0])
    print(df)
    print()
    with open(results_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            coordinates = prediction_utils.parse_coordinates(row["answer_coordinates"])
            all_coordinates.append(coordinates)
            answers = ', '.join([table[row + 1][col] for row, col in coordinates])
            position = int(row['position'])
            print(">", queries[position])
            print(answers)
    return all_coordinates


@dataclasses.dataclass
class TpuOptions:
    use_tpu: bool
    tpu_name: Optional[Text]
    tpu_zone: Optional[Text]
    gcp_project: Optional[Text]
    master: Optional[Text]
    num_tpu_cores: int
    iterations_per_loop: int


def _print(msg):
    print(msg)
    logging.info(msg)


def _warn(msg):
    print(f'Warning: {msg}')
    logging.warn(msg)


def convert_interactions_to_examples(tables_and_queries, converter):
    """Calls Tapas converter to convert interaction to example."""
    for idx, (table, queries) in enumerate(tables_and_queries):
        interaction = interaction_pb2.Interaction()
        for position, query in enumerate(queries):
            question = interaction.questions.add()
            question.original_text = query
            question.id = f"{idx}-0_{position}"
        for header in table[0]:
            interaction.table.columns.add().text = header
        for line in table[1:]:
            row = interaction.table.rows.add()
            for cell in line:
                row.cells.add().text = cell
        number_annotation_utils.add_numeric_values(interaction)
        for i in range(len(interaction.questions)):
            try:
                yield converter.convert(interaction, i)
            except ValueError as e:
                print(f"Can't convert interaction: {interaction.id} error: {e}")


def write_tf_example(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for example in examples:
            writer.write(example.SerializeToString())
