from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from src.types import CLIPInferenceOutput, CLIPTrainOutput, Embeddings, Logits
from src.utils.functions import normalize


class BaseModel(nn.Module):
    """Base model with property device."""

    @property
    def device(self) -> torch.device:
        """Device of model."""
        return next(self.parameters()).device


class VisionPartCLIP(BaseModel):
    """Vision part of clip with computing CLIP logits."""

    def __init__(
        self,
        model: nn.Module,
        output_model_shape: int,
        count_classes: int,
    ) -> None:
        """Init CV part of CLIP. 'model' is CNN or Transformer for feature extraction."""
        super().__init__()
        self.model = model
        self.clf = nn.Linear(in_features=output_model_shape, out_features=count_classes)

    def forward(self, image: Tensor, is_classification: bool = False) -> Tensor:
        """Forward method of vision part of CLIP."""
        output = self.model(image)
        if is_classification:
            output = self.clf(output)
        return output


class TextPartCLIP(BaseModel):
    """Text part of clip with computing CLIP logits."""

    CLS_IND = 0

    def __init__(
        self,
        model: nn.Module,
        output_model_shape: int,
        count_tokens: int,
        output_clip_shape: int,
    ) -> None:
        """Init text part of CLIP. 'model' is GPT2 model or another autoregressive model."""
        super().__init__()
        self.model = model
        self.lm = nn.Linear(in_features=output_model_shape, out_features=count_tokens)
        self.head = nn.Linear(
            in_features=output_model_shape, out_features=output_clip_shape
        )

    def forward(
        self,
        text: Tensor,
        perm_mask: Optional[Tensor] = None,
        target_mapping: Optional[Tensor] = None,
        is_masked_lm: bool = False,
    ) -> Tensor:
        """Forward method of text part of CLIP."""
        output = self.model(text, perm_mask=perm_mask, target_mapping=target_mapping)
        if is_masked_lm:
            output = self.lm(output).permute(0, 2, 1)
        else:
            output = self.head(output[:, self.CLS_IND, :])
        return output


class CLIP(BaseModel):
    """CLIP model with 2 parts - image part and text part."""

    def __init__(self, vision_part: VisionPartCLIP, text_part: TextPartCLIP) -> None:
        """Init of CLIP model."""
        super().__init__()
        self.vision_part = vision_part
        self.text_part = text_part
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_logit(self, embeddings: Embeddings) -> Logits:
        """Method for compute logits between image and text embeddings."""
        image_logit = self.logit_scale.exp() * embeddings.image @ embeddings.text.t()
        text_logit = image_logit.t()
        return Logits(image=image_logit, text=text_logit)

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        image_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
    ) -> CLIPTrainOutput:
        """Forward method of CLIP model."""
        if image_features is None:
            image_features = self.vision_part(image=image)
        image_embedding = normalize(image_features)
        if text_features is None:
            text_features = self.text_part(text=text)
        text_embedding = normalize(text_features)
        embeddings = Embeddings(image=image_embedding, text=text_embedding)
        logits = self.compute_logit(embeddings=embeddings)
        return CLIPTrainOutput(embeddings=embeddings, logits=logits)

    @torch.no_grad()
    def inference(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> CLIPInferenceOutput:
        """Inference of CLIP model."""
        output = self.forward(
            image=image,
            text=text,
            image_features=image_features,
            text_features=text_features,
        )
        classes = torch.argmax(output.logits.image, dim=-1)
        return CLIPInferenceOutput(classes=classes, embeddings=output.embeddings)
