from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

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


@dataclass
class DataConfig:
    train_clip_csv: Union[str, Path]
    valid_clip_csv: Union[str, Path]
    clip_name_image_column: str
    clip_name_text_column: str
    clip_batch_size_train: int
    clip_batch_size_valid: int
    train_image_classification_csv: Union[str, Path]
    valid_image_classification_csv: Union[str, Path]
    image_classification_name_image_column: str
    image_classification_name_label_column: str
    image_classification_batch_size_train: int
    image_classification_batch_size_valid: int
    image_classification_count_classes: int
    train_masked_lm_csv: Union[str, Path]
    valid_masked_lm_csv: Union[str, Path]
    masked_lm_name_text_column: str
    masked_lm_batch_size_train: int
    masked_lm_batch_size_valid: int
    cls_token_ids: int
    pad_token_ids: int
    tokens_max_len: int
    mask_token_idx: int
    masked_portion: float
    image_augmentations: Augmentations
    text_tokenizer_checkpoint: Union[str, Path]
    mask_text_transform: Callable
    num_workers: int = 2

    def __post_init__(self) -> None:
        assert 0 < self.masked_portion < 1
