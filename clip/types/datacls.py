from dataclasses import dataclass
from typing import Any, Dict

from torch import nn
from torch.utils.data import DataLoader

from .base import BaseOutput

#################################### COMMON ####################################


@dataclass
class Embeddings(BaseOutput):
    ...


@dataclass
class Logits(BaseOutput):
    ...


@dataclass
class CLIPOutput:
    embeddings: Embeddings
    logits: Logits


@dataclass
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader


@dataclass
class BatchSizes:
    train: int
    valid: int
    test: int


@dataclass
class SplitSizes:
    train: float
    valid: float
    test: float

    def __post_init__(self) -> None:
        assert self.train + self.valid + self.test == 1, "Sum of train/valid/test is equal 1"


@dataclass
class MultiTaskProportions:
    clip: float
    image: float
    text: float

    def __post_init__(self) -> None:
        assert self.clip >= 0
        assert self.image >= 0
        assert self.text >= 0


@dataclass
class MultiTaskCriterions:
    clip: nn.Module
    image: nn.Module
    text: nn.Module


@dataclass
class MultiTaskDataLoaders:
    clip: DataLoaders
    image: DataLoaders
    text: DataLoaders


################################### CONFIGS ####################################


@dataclass
class ImageDataConfig:
    dataframe: str
    batch_sizes: BatchSizes
    split_sizes: SplitSizes
    num_workers: int
    image_column: str
    label_column: str


@dataclass
class TextDataConfig:
    dataframe: str
    batch_sizes: BatchSizes
    split_sizes: SplitSizes
    num_workers: int
    text_column: str


@dataclass
class CLIPDataConfig:
    dataframe: str
    batch_sizes: BatchSizes
    split_sizes: SplitSizes
    num_workers: int
    image_column: str
    text_column: str


@dataclass
class DataConfig:
    tokenizer: str
    max_length: int
    image: ImageDataConfig
    text: TextDataConfig
    clip: CLIPDataConfig


@dataclass
class ModelConfig:
    image: Dict[str, Any]
    text: Dict[str, Any]


@dataclass
class TrainConfig:
    data: DataConfig
    model: ModelConfig


__all__ = [
    "Embeddings",
    "Logits",
    "CLIPOutput",
    "DataLoaders",
    "BatchSizes",
    "SplitSizes",
    "MultiTaskProportions",
    "MultiTaskCriterions",
    "MultiTaskDataLoaders",
    "TrainConfig",
]
