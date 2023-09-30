from pathlib import Path
from typing import Union

import torch
from torch.optim import lr_scheduler
from transformers import (
    BertConfig,
    BertModel,
    ConvNextConfig,
    ConvNextModel,
    ConvNextV2Config,
    ConvNextV2Model,
    DebertaConfig,
    DebertaModel,
    DebertaV2Config,
    DebertaV2Model,
    DistilBertConfig,
    DistilBertModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    SwinConfig,
    SwinModel,
    Swinv2Config,
    Swinv2Model,
)

Device = Union[str, torch.device]
PathLike = Union[str, Path]
Scheduler = Union[
    lr_scheduler._LRScheduler,  # pylint: disable=protected-access
    lr_scheduler.ReduceLROnPlateau,
]
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
ImageModel = Union[ConvNextModel, ConvNextV2Model, SwinModel, Swinv2Model]
ImageConfig = Union[ConvNextConfig, ConvNextV2Config, SwinConfig, Swinv2Config]
TextModel = Union[BertModel, DistilBertModel, DebertaModel, DebertaV2Model]
TextConfig = Union[BertConfig, DistilBertConfig, DebertaConfig, DebertaV2Config]

__all__ = ["Device", "PathLike", "Scheduler", "Tokenizer", "ImageModel", "ImageConfig", "TextModel", "TextConfig"]
