from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Embeddings:
    image: Optional[torch.Tensor] = None
    text: Optional[torch.Tensor] = None


@dataclass
class Logits:
    image: Optional[torch.Tensor] = None
    text: Optional[torch.Tensor] = None


@dataclass
class CLIPOutput:
    embeddings: Embeddings
    logits: Logits


@dataclass
class CLIPInferenceOutput:
    classes: torch.Tensor
    embeddings: Embeddings
