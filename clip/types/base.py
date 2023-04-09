import abc
import random

import torch
from torch import Tensor, nn


class BaseTextAugmentator:
    def __init__(self, p: float) -> None:
        assert 0 <= p <= 1, "probability must be in interval [0, 1]"
        self.p = p

    @abc.abstractmethod
    def apply(self, text: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    def transform(self, text: str) -> str:
        return text if random.random() > self.p else self.apply(text=text)


class BaseModule(nn.Module):  # pylint: disable=abstract-method
    @property
    def device(self) -> torch.device:
        """Device of model."""
        return next(self.parameters()).device


class ImageModel(BaseModule):
    @property
    @abc.abstractmethod
    def output_shape(self) -> int:
        raise NotImplementedError


class TextModel(BaseModule):
    @property
    @abc.abstractmethod
    def output_shape(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def mean_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        ...


__all__ = ["BaseTextAugmentator", "BaseModule", "ImageModel", "TextModel"]
