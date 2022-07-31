from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Embeddings:
    image: torch.Tensor
    text: torch.Tensor


@dataclass
class Logits:
    image: torch.Tensor
    text: torch.Tensor


@dataclass
class CLIPTrainOutput:
    embeddings: Embeddings
    logits: Logits


@dataclass
class CLIPInferenceOutput:
    classes: torch.Tensor
    embeddings: Embeddings


@dataclass
class ConvNeXtConfig:
    in_channels: int
    out_channels: int
    drop_path_rate: float
    depths: Optional[List[int]] = None
    dims: Optional[List[int]] = None


@dataclass
class XLNetConfig:
    num_layers: int
    vocab_size: int
    model_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    activation: str
    pad_idx: int
