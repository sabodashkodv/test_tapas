import pandas as pd
from tapas.utils import tasks
from test import run_question_answer, converter


path = 'data_sample/drivers.csv'
query = ["what were the drivers names?",
         "of these, which points did patrick carpentier and bruno junqueira score?",
         "who scored higher?"]
frame = pd.read_csv(path)
task = tasks.Task.SQA

result = run_question_answer(frame, query, task, converter)
