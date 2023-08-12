from dataclasses import dataclass

from torch.utils.data import DataLoader

from .base import BaseOutput


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


__all__ = ["Embeddings", "Logits", "CLIPOutput", "DataLoaders"]
