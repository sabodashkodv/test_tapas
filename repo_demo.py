import pandas as pd
from tapas.utils import tasks
from test import run_question_answer, converter


path = 'data_sample/repo.csv'
query = ["how many repositories?",
         "which programming language is the most used?",
         "which framework is implemented in rust?"]
frame = pd.read_csv(path)
task = tasks.Task.SQA

result = run_question_answer(frame, query, task, converter)
