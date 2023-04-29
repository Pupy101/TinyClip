import abc
import random
from dataclasses import dataclass
from typing import Set

import nltk
import torch
from torch import Tensor, nn


@dataclass
class BaseOutput:
    image: Tensor
    text: Tensor


class BaseModule(nn.Module):  # pylint: disable=abstract-method
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class BaseTextAugmentator:
    INSTALLED_REQUIREMENTS: bool = False  # parameter for download requirements once

    def __init__(self, p: float) -> None:
        assert 0 <= p <= 1, "probability must be in interval [0, 1]"
        self.p = p
        self.nltk_requirements: Set[str] = set()
        self.add_requirements()

    def add_nltk_requirements(self, *args: str) -> None:
        self.nltk_requirements.update(args)

    def install_requirements(self) -> None:
        if not self.__class__.INSTALLED_REQUIREMENTS:
            for requirement in self.nltk_requirements:
                nltk.download(requirement)
            self.__class__.INSTALLED_REQUIREMENTS = True

    @abc.abstractmethod
    def apply(self, text: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def add_requirements(self) -> None:
        raise NotImplementedError

    def transform(self, text: str) -> str:
        return text if random.random() > self.p else self.apply(text=text)


__all__ = ["BaseOutput", "BaseModule", "BaseTextAugmentator"]
