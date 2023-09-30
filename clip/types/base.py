import abc
import random
from dataclasses import dataclass
from typing import List, Set

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
    def __init__(self, p: float) -> None:
        assert 0 <= p <= 1, "probability must be in interval [0, 1]"
        self.p = p

    @abc.abstractmethod
    def apply(self, text: str) -> str:
        raise NotImplementedError

    def transform(self, text: str) -> str:
        return text if random.random() > self.p else self.apply(text=text)


class BaseNLTKTextAugmentator(BaseTextAugmentator):
    def __init__(self, p: float) -> None:
        super().__init__(p=p)

    @property
    @abc.abstractmethod
    def nltk_requirements(self) -> Set[str]:
        raise NotImplementedError


class BaseComposeTextAugmentator(BaseTextAugmentator):
    NLTK_REQUIREMENTS: Set[str] = set()
    INSTALLED_REQUIREMENTS: bool = False

    def __init__(self, augmentators: List[BaseTextAugmentator]) -> None:
        self.augmentators = augmentators
        super().__init__(p=1.0)
        for augmentator in self.augmentators:
            if isinstance(augmentator, BaseNLTKTextAugmentator):
                type(self).NLTK_REQUIREMENTS.update(augmentator.nltk_requirements)

    def apply(self, text: str) -> str:
        if not self.INSTALLED_REQUIREMENTS and self.NLTK_REQUIREMENTS:
            for requirement in self.NLTK_REQUIREMENTS:
                nltk.download(requirement, quiet=True)
            type(self).INSTALLED_REQUIREMENTS = True
        for augmentator in self.augmentators:
            text = augmentator.transform(text=text)
        return text


__all__ = [
    "BaseOutput",
    "BaseModule",
    "BaseTextAugmentator",
    "BaseNLTKTextAugmentator",
    "BaseComposeTextAugmentator",
]
