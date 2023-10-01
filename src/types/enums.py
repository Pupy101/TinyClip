from enum import Enum


class DatasetMark(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


__all__ = ["DatasetMark"]
