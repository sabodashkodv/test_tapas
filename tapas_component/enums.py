import enum


class Task(enum.Enum):
    """Fine-tuning tasks supported by Tapas."""
    SQA = 0
    WTQ = 1
    WIKISQL = 2
    WIKISQL_SUPERVISED = 3
    TABFACT = 4


class Mode(enum.Enum):
    CREATE_DATA = 1
    TRAIN = 2
    PREDICT_AND_EVALUATE = 3
    EVALUATE = 4
    PREDICT = 5


class TestSet(enum.Enum):
    DEV = 1
    TEST = 2
