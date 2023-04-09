import random
from typing import List, Set

import albumentations as A
import nltk
from albumentations.pytorch import ToTensorV2
from nltk.tokenize import word_tokenize

from clip.types import BaseTextAugmentator, DatasetType

from .utils import get_synonyms


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
        augmentations.extend(
            [
                A.SmallestMaxSize(max_size=250),
                A.CenterCrop(height=224, width=224),
            ]
        )
    else:
        raise ValueError(
            f"Strange transform_type for 'create_image_augmentations': {transform_type}"
        )

    augmentations.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(augmentations)


class ComposeAugmentator:
    def __init__(self, augmentators: List[BaseTextAugmentator]) -> None:
        self.augmentators = augmentators

        for augmentator in self.augmentators:
            augmentator.prepare()

    def transform(self, text: str) -> str:
        for augmentator in self.augmentators:
            text = augmentator.transform(text=text)
        return text

    def __call__(self, text: str) -> str:
        return self.transform(text=text)


class SynonymReplacementAugmentator(BaseTextAugmentator):
    def __init__(self, p: float, p_replace: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_replace <= 1, "probability must be in interval [0, 1]"
        self.p_replace = p_replace

    def prepare(self) -> None:
        nltk.download("punkt")
        nltk.download("wordnet")

    def apply(self, text: str) -> str:
        new_sentence = []
        for word in word_tokenize(text):
            if random.random() < self.p_replace:
                synonyms = get_synonyms(word)
                if synonyms:
                    new_sentence.append(random.choice(synonyms))
                else:
                    new_sentence.append(word)
            else:
                new_sentence.append(word)
        return " ".join(new_sentence)


class RandomInsertAugmentator(BaseTextAugmentator):
    def __init__(self, p: float, p_insert: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_insert <= 1, "probability must be in interval [0, 1]"
        self.p_insert = p_insert

    def prepare(self) -> None:
        nltk.download("punkt")
        nltk.download("wordnet")

    def apply(self, text: str) -> str:
        new_words = []
        for word in word_tokenize(text):
            new_words.append(word)
            if random.random() < self.p_insert:
                synonyms = get_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(synonyms))
        return " ".join(new_words)


class CharSwapAugmentator(BaseTextAugmentator):
    def __init__(self, p: float, p_swap: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_swap <= 1, "probability must be in interval [0, 1]"
        self.p_swap = p_swap

    def prepare(self) -> None:
        nltk.download("punkt")

    def apply(self, text: str) -> str:
        swapped_words: List[str] = []
        for word in word_tokenize(text):
            if len(word) > 1 and random.random() < self.p_swap:
                chars = list(word)
                random.shuffle(chars)
                word = "".join(chars)
            swapped_words.append(word)

        return " ".join(swapped_words)


class WordsReorderingAugmentator(BaseTextAugmentator):
    def __init__(self, p: float, p_reordering: float) -> None:
        super().__init__(p=p)
        assert 0 <= p_reordering <= 1, "probability must be in interval [0, 1]"
        self.p_reordering = p_reordering

    def prepare(self) -> None:
        nltk.download("punkt")

    def apply(self, text: str) -> str:
        reordering_indexes: Set[int] = set()
        reordering_words: List[str] = []
        for i, word in enumerate(word_tokenize(text)):
            if random.random() < self.p_reordering:
                reordering_indexes.add(i)
                reordering_words.append(word)
        random.shuffle(reordering_words)
        new_words: List[str] = []
        for i, word in enumerate(word_tokenize(text)):
            if i in reordering_indexes:
                new_words.append(reordering_words.pop())
            else:
                new_words.append(word)
        return " ".join(new_words)


def create_text_augs(transform_type: str) -> ComposeAugmentator:
    augmentations: List[BaseTextAugmentator] = []
    if transform_type == DatasetType.TRAIN.value:
        augmentations.extend(
            [
                SynonymReplacementAugmentator(p=0.5, p_replace=0.4),
                RandomInsertAugmentator(p=0.5, p_insert=0.3),
                CharSwapAugmentator(p=0.5, p_swap=0.2),
                WordsReorderingAugmentator(p=0.5, p_reordering=0.3),
            ]
        )
    elif transform_type in {DatasetType.VALID.value, DatasetType.TEST.value}:
        augmentations.extend(
            [
                SynonymReplacementAugmentator(p=0.5, p_replace=0.4),
            ]
        )
    else:
        raise ValueError(f"Strange transform_type for 'create_augmentations': {transform_type}")
    return ComposeAugmentator(augmentations)
