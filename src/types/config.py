from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Type, Union

import torch
from torch import nn, optim

from .data import Augmentations
from .train import MultiTaskProportion


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


@dataclass
class TrainConfig:
    n_epochs: int
    optimizer: Type[optim.Optimizer]
    criterion_clip: nn.Module
    criterion_image: nn.Module
    criterion_text: nn.Module
    device: Union[str, torch.device]
    save_dir: Union[str, Path]
    coefficients: MultiTaskProportion
    optimizer_params: dict = field(default_factory=dict)
    scheduler: Optional[Type[optim.lr_scheduler._LRScheduler]] = None
    scheduler_params: dict = field(default_factory=dict)
    accumulation_steps: int = 1
    seed: int = 1234

    def __post_init__(self) -> None:
        assert self.accumulation_steps >= 1
        assert self.n_epochs >= 1
