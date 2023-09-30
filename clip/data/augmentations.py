import random
import re
from typing import List, Set

import albumentations as A
from albumentations.pytorch import ToTensorV2
from nltk.tokenize import word_tokenize

from clip.data.utils import get_synonyms
from clip.types import BaseComposeTextAugmentator, BaseNLTKTextAugmentator, BaseTextAugmentator, DatasetType


def create_image_augs(transform_type: str) -> A.Compose:
    augmentations: List[A.BasicTransform] = []

    if transform_type == DatasetType.TRAIN.value:
        augmentations.extend(
            [
                A.SmallestMaxSize(max_size=250),
                A.RandomCrop(height=224, width=224),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.CLAHE(),
                A.GaussNoise(),
                A.CoarseDropout(),
                A.ChannelDropout(),
                A.Rotate(limit=15),
            ]
        )
    elif transform_type in {DatasetType.VALID.value, DatasetType.TEST.value}:
        augmentations.extend([A.SmallestMaxSize(max_size=250), A.CenterCrop(height=224, width=224)])
    else:
        raise ValueError(f"Strange transform_type for 'create_image_augmentations': {transform_type}")

    augmentations.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(augmentations)


def create_text_augs(transform_type: str) -> "ComposeAugmentator":
    augmentations: List[BaseTextAugmentator] = []

    if transform_type == DatasetType.TRAIN.value:
        augmentations.extend(
            [
                SynonymReplacementNLTKAugmentator(p=0.5, p_replace=0.4),
                RandomInsertNLTKAugmentator(p=0.5, p_insert=0.4),
                CharSwapNLTKAugmentator(p=0.3, p_swap=0.1),
            ]
        )
    elif transform_type in {DatasetType.VALID.value, DatasetType.TEST.value}:
        pass  # doesn't use any transform
    else:
        raise ValueError(f"Strange transform_type for 'create_augmentations': {transform_type}")

    return ComposeAugmentator(augmentations)


class ComposeAugmentator(BaseComposeTextAugmentator):
    def __call__(self, text: str) -> str:
        return self.transform(text=text)


class SynonymReplacementNLTKAugmentator(BaseNLTKTextAugmentator):
    def __init__(self, p: float, p_replace: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_replace <= 1, "probability must be in interval [0, 1]"
        self.p_replace = p_replace

    @property
    def nltk_requirements(self) -> Set[str]:
        return {"punkt", "wordnet"}

    def apply(self, text: str) -> str:
        for word in word_tokenize(text):
            if re.match(r"[\w]{4,}", word) and random.random() < self.p_replace:
                synonyms = get_synonyms(word)
                if synonyms:
                    text = re.sub(word, random.choice(synonyms), text)
        return text


class RandomInsertNLTKAugmentator(BaseNLTKTextAugmentator):
    def __init__(self, p: float, p_insert: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_insert <= 1, "probability must be in interval [0, 1]"
        self.p_insert = p_insert

    @property
    def nltk_requirements(self) -> Set[str]:
        return {"punkt", "wordnet"}

    def apply(self, text: str) -> str:
        for word in word_tokenize(text):
            if re.match(r"[\w]{4,}", word) and random.random() < self.p_insert:
                synonyms = get_synonyms(word)
                if synonyms:
                    text = re.sub(f"({word})", "\\1 " + random.choice(synonyms), text)
        return text


class CharSwapNLTKAugmentator(BaseNLTKTextAugmentator):
    def __init__(self, p: float, p_swap: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_swap <= 1, "probability must be in interval [0, 1]"
        self.p_swap = p_swap

    @property
    def nltk_requirements(self) -> Set[str]:
        return {"punkt"}

    def apply(self, text: str) -> str:
        changed_words: Set[str] = set()
        for word in word_tokenize(text):
            if word not in changed_words and re.match(r"[\w]{4,}", word) and random.random() < self.p_swap:
                chars = list(word)
                random.shuffle(chars)
                new_word = "".join(chars)
                text = re.sub(word, new_word, text)
            changed_words.add(word)
        return text
