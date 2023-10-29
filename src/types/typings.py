from pathlib import Path
from typing import Union

import torch
from torch.optim import Optimizer, lr_scheduler  # pylint: disable=unused-import
from transformers import AutoImageProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast

Device = Union[str, torch.device]
PathLike = Union[str, Path]
Scheduler = Union[lr_scheduler.LRScheduler, lr_scheduler.ReduceLROnPlateau]
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Processor = AutoImageProcessor
