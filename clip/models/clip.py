from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from clip.types import (
    BaseModule,
    CLIPInferenceOutput,
    CLIPTrainOutput,
    Embeddings,
    ImageModel,
    Logits,
    TextModel,
)


class ImagePartCLIP(BaseModule):
    def __init__(self, model: ImageModel, count_classes: int) -> None:
        super().__init__()
        self.model = model
        self.clf = nn.Linear(in_features=model.output_shape, out_features=count_classes)

    def forward(self, image: Tensor, classification: bool = False) -> Tensor:
        output: Tensor
        output = self.model(image)
        if classification:
            output = self.clf(output)
        return output


class TextPartCLIP(BaseModule):
    def __init__(self, model: TextModel) -> None:
        super().__init__()
        self.model = model
        self.head = nn.Linear(in_features=model.output_shape, out_features=model.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        masked_lm: bool = False,
    ) -> Tensor:
        output: Tensor
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if masked_lm:
            output = self.head(output)
            # change for cross entropy [b_s, s_l, m_d] -> [b_s, m_d, s_l]
            return output.permute(0, 2, 1)
        if attention_mask is not None:
            return self.model.mean_pooling(embeddings=output, attention_mask=attention_mask)
        return torch.mean(output, dim=1)


class CLIP(BaseModule):
    def __init__(self, image_part: ImagePartCLIP, text_part: TextPartCLIP) -> None:
        super().__init__()
        self.image_part = image_part
        self.text_part = text_part
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_logit(self, image: Tensor, text: Tensor) -> Logits:
        image_logit = self.logit_scale.exp() * image @ text.t()
        text_logit = image_logit.t()
        return Logits(image=image_logit, text=text_logit)

    def forward(  # pylint: disable=too-many-arguments
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        image_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ) -> CLIPTrainOutput:
        if image is None and image_features is None:
            raise ValueError("Set image or image_features")
        if text is None and text_features is None:
            raise ValueError("Set text or text_features")

        if image_features is None:
            image_features = self.image_part(image=image)
        if text_features is None:
            text_features = self.text_part(text=text, attention_mask=text_attention_mask)
        embeddings = Embeddings(image=normalize(image_features), text=normalize(text_features))
        logits = self.compute_logit(image=embeddings.image, text=embeddings.text)
        return CLIPTrainOutput(embeddings=embeddings, logits=logits)

    @torch.no_grad()
    def inference(  # pylint: disable=too-many-arguments
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ) -> CLIPInferenceOutput:
        output = self.forward(
            image=image,
            text=text,
            image_features=image_features,
            text_features=text_features,
            text_attention_mask=text_attention_mask,
        )
        classes = torch.argmax(output.logits.image, dim=-1)
        return CLIPInferenceOutput(classes=classes, embeddings=output.embeddings)


__all__ = ["ImagePartCLIP", "TextPartCLIP", "CLIP"]
