from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import (
    BertConfig,
    ConvNextConfig,
    ConvNextV2Config,
    DebertaConfig,
    DebertaV2Config,
    DistilBertConfig,
    SwinConfig,
    Swinv2Config,
)

from clip.types import (
    BaseModule,
    CLIPOutput,
    Embeddings,
    ImageConfig,
    ImageModel,
    ImageModelType,
    Logits,
    TextConfig,
    TextModel,
    TextModelType,
    check_enum,
)

from .image.convnext import create_convnext, create_convnext_v2
from .image.swin import create_swin, create_swin_v2
from .text.bert import create_bert, create_distil_bert
from .text.deberta import create_deberta, create_deberta_v2


class ImagePart(BaseModule):
    def __init__(self, model_type: str, **model_params: Any) -> None:
        super().__init__()
        check_enum(model_type, ImageModelType)
        self.config, self.model = self.init_model(model_type=model_type, **model_params)
        if isinstance(self.config, (ConvNextConfig, ConvNextV2Config)):
            self.out_shape = self.config.hidden_sizes[-1]
        elif isinstance(self.config, (SwinConfig, Swinv2Config)):
            self.out_shape = self.config.embed_dim * 8
        else:
            raise TypeError(f"Strange config type: {type(self.config)}")

    @staticmethod
    def init_model(model_type: str, **model_params: Any) -> Tuple[ImageConfig, ImageModel]:
        if model_type == ImageModelType.CONVNEXT.value:
            return create_convnext(**model_params)
        if model_type == ImageModelType.CONVNEXT_V2.value:
            return create_convnext_v2(**model_params)
        if model_type == ImageModelType.SWIN.value:
            return create_swin(**model_params)
        if model_type == ImageModelType.SWINV2.value:
            return create_swin_v2(**model_params)
        raise ValueError(f"Strange model type: {model_type}")

    def forward(self, images: Tensor) -> Tensor:
        return self.model.forward(images).pooler_output


class TextPart(BaseModule):
    def __init__(self, model_type: str, **model_params: Any) -> None:
        super().__init__()
        check_enum(model_type, TextModelType)
        self.config, self.model = self.init_model(model_type=model_type, **model_params)
        if isinstance(self.config, (BertConfig, DistilBertConfig, DebertaConfig, DebertaV2Config)):
            self.out_shape = self.config.hidden_size
        else:
            raise TypeError(f"Strange config type: {type(self.config)}")

    @staticmethod
    def init_model(model_type: str, **model_params: Any) -> Tuple[TextConfig, TextModel]:
        if model_type == TextModelType.BERT.value:
            return create_bert(**model_params)
        if model_type == TextModelType.DISTILBERT.value:
            return create_distil_bert(**model_params)
        if model_type == TextModelType.DEBERTA.value:
            return create_deberta(**model_params)
        if model_type == TextModelType.DEBERTA_V2.value:
            return create_deberta_v2(**model_params)
        raise ValueError(f"Strange model type: {model_type}")

    @staticmethod
    def mean_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        weighted_embeddings = embeddings * attention_mask.unsqueeze(-1)
        mean_embeddings = torch.mean(weighted_embeddings, dim=1)
        return F.normalize(mean_embeddings, dim=1)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if attention_mask is not None:
            return self.mean_pooling(embeddings=output, attention_mask=attention_mask)
        return torch.mean(output, dim=1)


class TextPartMLM(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.text_part = TextPart(*args, **kwargs)
        self.lm_head = nn.Linear(self.text_part.config.hidden_size, self.text_part.config.vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        output = self.text_part.forward(input_ids=input_ids, attention_mask=attention_mask)
        return self.lm_head.forward(output).permute(0, 2, 1)


class CLIP(BaseModule):
    def __init__(self, image_part: ImagePart, text_part: TextPart) -> None:
        super().__init__()
        assert image_part.out_shape == text_part.out_shape
        self.image_part = image_part
        self.text_part = text_part
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_logit(self, image: Tensor, text: Tensor) -> Logits:
        image_logit = self.logit_scale.exp() * image @ text.t()
        text_logit = image_logit.t()
        return Logits(image=image_logit, text=text_logit)

    def forward(
        self,
        images: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
    ) -> CLIPOutput:
        if images is None and image_features is None:
            raise ValueError("Set images or images_features")
        if input_ids is None and text_features is None:
            raise ValueError("Set input_ids or text_features")

        if image_features is None and images is not None:
            image_features = self.image_part.forward(images=images)
        if text_features is None and input_ids is not None:
            text_features = self.text_part.forward(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = Embeddings(image=F.normalize(image_features), text=F.normalize(text_features))  # type: ignore
        logits = self.compute_logit(image=embeddings.image, text=embeddings.text)
        return CLIPOutput(embeddings=embeddings, logits=logits)
