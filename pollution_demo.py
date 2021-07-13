import pandas as pd
from tapas.utils import tasks
from test import run_question_answer, converter


path = 'data_sample/pollution.csv'
query = ["What is the highest pollution?",
         "What is the maximum temperature?"]
frame = pd.read_csv(path).head(100)
task = tasks.Task.SQA

result = run_question_answer(frame, query, task, converter)
