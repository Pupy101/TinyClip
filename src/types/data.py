from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader


@dataclass
class MultiTaskDataLoaders:
    clip: DataLoaders
    image: DataLoaders
    text: DataLoaders


@dataclass
class Augmentations:
    train: Any
    validation: Any
