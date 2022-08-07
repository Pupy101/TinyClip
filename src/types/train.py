from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn, optim


@dataclass
class MultiTaskProportion:
    clip: float
    image: float
    text: float

    def __post_init__(self) -> None:
        assert 0 <= self.clip
        assert 0 <= self.image
        assert 0 <= self.text


@dataclass
class TrainConfig:
    train_clip: bool
    train_image: bool
    train_text: bool
    n_epochs: int
    count_accumulated_batches: int
    optimizer: optim.Optimizer
    criterion_clip: nn.Module
    criterion_image: nn.Module
    criterion_text: nn.Module
    device: Union[str, torch.device]
    save_dir: Union[str, Path]
    coefficients: MultiTaskProportion
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    seed: int = 1234

    def __post_init__(self) -> None:
        assert (
            self.count_accumulated_batches >= 1
        ), "Set 1 or more count_accumulated_batches"
        assert self.n_epochs >= 1, "Set 1 or more count training epochs"
        assert (
            self.train_clip or self.train_image or self.train_text
        ), "Set one of or all training type 'train_clip', 'train_image', 'train_text'"
