from typing import Any, Dict, Optional, Tuple

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

from .bert import create_bert, create_distil_bert, pretrained_bert, pretrained_distil_bert
from .convnext import create_convnext, create_convnext_v2, pretrained_convnext, pretrained_convnext_v2
from .deberta import create_deberta, create_deberta_v2, pretrained_deberta, pretrained_deberta_v2
from .swin import create_swin, create_swin_v2, pretrained_swin, pretrained_swin_v2


class ImagePartCLIP(BaseModule):
    def __init__(self, model_type: str, count_classes: int, pretrained: bool = False, **model_params: Any) -> None:
        super().__init__()
        check_enum(model_type, ImageModelType)
        self.count_classes = count_classes
        self.config, self.model = self.init_model(model_type=model_type, pretrained=pretrained, **model_params)
        if isinstance(self.config, (ConvNextConfig, ConvNextV2Config)):
            self.out_shape = self.config.hidden_sizes[-1]
        elif isinstance(self.config, (SwinConfig, Swinv2Config)):
            self.out_shape = self.config.embed_dim * 8
        else:
            raise TypeError(f"Strange config type: {type(self.config)}")
        self.classificator = nn.Linear(in_features=self.out_shape, out_features=count_classes)

    @staticmethod
    def init_model(  # pylint: disable=too-many-return-statements
        model_type: str,
        pretrained: bool,
        **model_params: Any,
    ) -> Tuple[ImageConfig, ImageModel]:
        if pretrained and model_type == ImageModelType.CONVNEXT.value:
            return pretrained_convnext(**model_params)
        if not pretrained and model_type == ImageModelType.CONVNEXT.value:
            return create_convnext(**model_params)
        if pretrained and model_type == ImageModelType.CONVNEXTV2.value:
            return pretrained_convnext_v2(**model_params)
        if not pretrained and model_type == ImageModelType.CONVNEXTV2.value:
            return create_convnext_v2(**model_params)
        if pretrained and model_type == ImageModelType.SWIN.value:
            return pretrained_swin(**model_params)
        if not pretrained and model_type == ImageModelType.SWIN.value:
            return create_swin(**model_params)
        if pretrained and model_type == ImageModelType.SWINV2.value:
            return pretrained_swin_v2(**model_params)
        if not pretrained and model_type == ImageModelType.SWINV2.value:
            return create_swin_v2(**model_params)
        raise ValueError(f"Strange model type: {model_type}")

    def forward(self, images: Tensor, classification: bool = False) -> Tensor:
        output = self.model.forward(images).pooler_output
        if classification:
            output = self.classificator(output)
        return output


class TextPartCLIP(BaseModule):
    def __init__(self, model_type: str, pretrained: bool = False, **model_params: Any) -> None:
        super().__init__()
        check_enum(model_type, TextModelType)
        self.config, self.model = self.init_model(model_type=model_type, pretrained=pretrained, **model_params)
        if isinstance(self.config, (BertConfig, DistilBertConfig, DebertaConfig, DebertaV2Config)):
            self.out_shape = self.config.hidden_size
            self.vocab_size = self.config.vocab_size
        else:
            raise TypeError(f"Strange config type: {type(self.config)}")
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.vocab_size)

    @staticmethod
    def init_model(  # pylint: disable=too-many-return-statements
        model_type: str,
        pretrained: bool,
        **model_params: Any,
    ) -> Tuple[TextConfig, TextModel]:
        if pretrained and model_type == TextModelType.BERT.value:
            return pretrained_bert(**model_params)
        if not pretrained and model_type == TextModelType.BERT.value:
            return create_bert(**model_params)
        if pretrained and model_type == TextModelType.DISTILBERT.value:
            return pretrained_distil_bert(**model_params)
        if not pretrained and model_type == TextModelType.DISTILBERT.value:
            return create_distil_bert(**model_params)
        if pretrained and model_type == TextModelType.DEBERTA.value:
            return pretrained_deberta(**model_params)
        if not pretrained and model_type == TextModelType.DEBERTA.value:
            return create_deberta(**model_params)
        if pretrained and model_type == TextModelType.DEBERTAV2.value:
            return pretrained_deberta_v2(**model_params)
        if not pretrained and model_type == TextModelType.DEBERTAV2.value:
            return create_deberta_v2(**model_params)
        raise ValueError(f"Strange model type: {model_type}")

    @staticmethod
    def mean_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        weighted_embeddings = embeddings * attention_mask.unsqueeze(-1)
        mean_embeddings = torch.mean(weighted_embeddings, dim=1)
        return F.normalize(mean_embeddings, dim=1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        masked_lm: bool = False,
    ) -> Tensor:
        output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if masked_lm:
            return self.lm_head.forward(output).permute(0, 2, 1)
        if attention_mask is not None:
            return self.mean_pooling(embeddings=output, attention_mask=attention_mask)
        return torch.mean(output, dim=1)


class CLIP(BaseModule):
    def __init__(self, image_part: ImagePartCLIP, text_part: TextPartCLIP) -> None:
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
        image_kwargs: Optional[Dict[str, Tensor]] = None,
        text_kwargs: Optional[Dict[str, Tensor]] = None,
        image_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
    ) -> CLIPOutput:
        if image_kwargs is None and image_features is None:
            raise ValueError("Set image or images_features")
        if text_kwargs is None and text_features is None:
            raise ValueError("Set text or text_features")

        if image_features is None and image_kwargs is not None:
            image_features = self.image_part(**image_kwargs)
        if text_features is None and text_kwargs:
            text_features = self.text_part(**text_kwargs)
        embeddings = Embeddings(image=F.normalize(image_features), text=F.normalize(text_features))  # type: ignore
        logits = self.compute_logit(image=embeddings.image, text=embeddings.text)
        return CLIPOutput(embeddings=embeddings, logits=logits)
