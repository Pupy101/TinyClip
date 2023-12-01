from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data.common import TOKENIZER_KWARGS
from src.types import PathLike

from .models.clip import Clip


class ClipInference:
    def __init__(self, img_pretrained: str, txt_pretrained: str, embedding_dim: int) -> None:
        self.clip = Clip(
            img=AutoModelForImageClassification.from_pretrained(
                img_pretrained, num_labels=embedding_dim, ignore_mismatched_sizes=True
            ),
            txt=AutoModelForSequenceClassification.from_pretrained(txt_pretrained, num_labels=embedding_dim),
        )
        self.preprocessor = AutoImageProcessor.from_pretrained(img_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(txt_pretrained)

    def load(
        self,
        path: PathLike,
        lightning: bool = False,
        map_location: Optional[Union[torch.device, str, Dict[str, str]]] = None,
    ) -> None:
        state_dict: dict = torch.load(path, map_location=map_location)
        if lightning:
            assert (
                "state_dict" in state_dict
                and isinstance(state_dict["state_dict"], dict)
                and any(k.startswith("clip.") for k in state_dict["state_dict"])
            )
            state_dict = {
                k.replace("clip.", "", 1): v for k, v in state_dict["state_dict"].items() if k.startswith("clip.")
            }
        return self.clip.load_state_dict(state_dict=state_dict)

    @torch.no_grad()
    def forward_img(self, img: List[Image.Image]) -> torch.Tensor:
        batch = self.preprocessor(images=img, return_tensors="pt").to(self.clip.img.device)
        return self.clip.normalize(self.clip.img(**batch).logits)

    @torch.no_grad()
    def forward_txt(self, txt: List[str], max_length: int = 120) -> torch.Tensor:
        batch = self.tokenizer(text=txt, max_length=max_length, **TOKENIZER_KWARGS).to(self.clip.txt.device)
        return self.clip.normalize(self.clip.txt(**batch).logits)

    @torch.no_grad()
    def predict(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        img = img.to(self.clip.scale.device)
        txt = txt.to(self.clip.scale.device)
        logits, _ = self.clip.forward(img=img, txt=txt)
        return torch.argmax(logits, dim=-1)
