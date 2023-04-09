from pathlib import Path
from typing import Union

import torch
from torch.optim import lr_scheduler
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Device = Union[str, torch.device]
PathLike = Union[str, Path]
Scheduler = Union[
    lr_scheduler._LRScheduler,  # pylint: disable=protected-access
    lr_scheduler.ReduceLROnPlateau,
]
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

__all__ = ["Device", "PathLike", "Scheduler", "Tokenizer"]
