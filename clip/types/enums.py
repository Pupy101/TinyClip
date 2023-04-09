from enum import Enum


class DatasetType(Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


__all__ = ["DatasetType"]
