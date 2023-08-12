from enum import Enum
from typing import Any, Type


def check_enum(value: Any, enum: Type[Enum]) -> None:
    enum_values = {_.value for _ in enum}
    if value not in enum_values:
        raise ValueError(f"Not find value: {value} in enum values: {enum_values}")


class ImageModelType(Enum):
    CONVNEXT = "convnext"
    CONVNEXT_V2 = "convnext_v2"
    SWIN = "swin"
    SWINV2 = "swin_v2"


class TextModelType(Enum):
    BERT = "bert"
    DISTILBERT = "distilbert"
    DEBERTA = "deberta"
    DEBERTA_V2 = "deberta_v2"


class DatasetType(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class TrainMode(Enum):
    TEXT = "text"
    IMAGE = "image"
    CLIP = "clip"


__all__ = ["check_enum", "ImageModelType", "TextModelType", "DatasetType", "TrainMode"]
