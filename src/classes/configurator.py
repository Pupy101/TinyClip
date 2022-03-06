from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Type

import torch

from pandas import DataFrame
from torch import nn, optim
from torch.utils.data import DataLoader

from ..model import CLIP


@dataclass
class TrainParameters:
    accumulation: int
    criterion: nn.Module
    device: torch.device
    loaders: Dict[str, DataLoader]
    model: CLIP
    n_epoch: int
    optimizer: optim.Optimizer
    save_dir: Path
    scheduler: Type[optim.lr_scheduler.StepLR]


@dataclass
class EvaluationParameters:
    classes: List[str]
    csv: DataFrame
    device: torch.device
    loaders: Dict[str, DataLoader]
    model: CLIP
    target_dir: Path
    tokenizer: Callable
