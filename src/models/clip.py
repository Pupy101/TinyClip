from typing import Tuple

from numpy import log
from torch import Tensor, nn, ones
from torch.nn import functional as F


class Clip(nn.Module):
    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(ones([]) * log(1 / 0.07))

    def forward(self, image_embeddings: Tensor, text_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        image_logits = self.logit_scale.exp() * image_embeddings @ text_embeddings.t()
        text_logits = image_logits.t()
        return image_logits, text_logits

    def normalize(self, embeddings: Tensor) -> Tensor:
        return F.normalize(embeddings, p=2, dim=-1)
