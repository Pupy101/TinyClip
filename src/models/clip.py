from typing import Tuple

from numpy import log
from torch import Tensor, nn, ones
from torch.nn import functional as F


class Clip(nn.Module):
    def __init__(self, img: nn.Module, txt: nn.Module) -> None:
        """
        Args:
            img: image encoder
            txt: text encoder
        """
        super().__init__()
        self.img = img
        self.txt = txt
        self.scale = nn.Parameter(ones([]) * log(1 / 0.07))

    def forward(self, img: Tensor, txt: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            img: image embedding
            txt: text embedding
        """
        img_logit = self.scale.exp() * img @ txt.t()
        txt_logit = img_logit.t()
        return img_logit, txt_logit

    def normalize(self, emb: Tensor) -> Tensor:
        return F.normalize(emb, p=2, dim=-1)
