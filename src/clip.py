from typing import Optional, Tuple

import torch
import numpy as np

from torch import nn
from torch.nn.functional import normalize


class VisionPartCLIP(nn.Module):
    """
    Vision part of clip with scale logit
    """
    def __init__(self, vision_model: nn.Module) -> None:
        """
        Method for init vision part CLIP

        :param vision_model: CNN model for getting image embedding
        """
        super().__init__()
        self.vision_model: nn.Module = vision_model
        self.logit_scale: nn.Parameter = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )

    def forward(
            self,
            image: torch.Tensor,
            text_embedding: torch.Tensor,
            image_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method of vision part with computing image and text embedding
        cosine similarity

        :param image: input image
        :param text_embedding: normalized text embedding
        :param image_features: vector representation of image from vision model
        :return: image and text logits, image embedding
        """
        if image_features is None:
            image_features = self.vision_model(image)
        image_embedding = normalize(image_features)
        logit_scale = self.logit_scale.exp()
        image_logit = logit_scale * image_embedding@text_embedding.t()
        text_logit = image_logit.t()
        return image_logit, text_logit, image_embedding


class CLIP(nn.Module):
    """
    CLIP model with 2 parts:
    image part - CNN and text part - Transformer
    """
    def __init__(
            self,
            vision_clip_part: VisionPartCLIP,
            text_model: nn.Module,
    ) -> None:
        """
        Method for init CLIP

        :param vision_clip_part: vision part of clip
        :param text_model: Transform for embedding text
        """
        super().__init__()
        self.vision_clip_part = vision_clip_part
        self.text_model = text_model

    def forward(
            self,
            image: torch.Tensor,
            text: torch.Tensor,
            image_features: Optional[torch.Tensor] = None,
            text_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method CLIP

        :param image: input image
        :param text: input text description
        :param image_features: vector representation of image from vision model
        :param text_features: vector representation of text from text model
        :return: image and text logits, (image_embedding, text_embedding)
        """
        if text_features is None:
            text_features = self.text_model(text)
        text_embedding = normalize(text_features)
        image_logit, text_logit, image_embedding = self.vision_clip_part(
            image, text_embedding, image_features
        )
        return image_logit, text_logit, (image_embedding, text_embedding)

    @torch.no_grad()
    def inference(
            self,
            image: torch.Tensor,
            text: torch.Tensor,
            image_features: Optional[torch.Tensor] = None,
            text_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inference forward CLIP

        :param image: input image
        :param text: input text classes
        :param image_features: images embedding vectors
        :param text_features: text embedding vectors
        :return: classes of input images
        """
        image_logit, text_logit, (image_embedding, text_embedding) = (
            self.forward(image, text, image_features, text_features)
        )
        classes = torch.argmax(image_logit, dim=1)
        return classes, (image_embedding, text_embedding)

    @property
    def device(self):
        return next(iter(self.parameters())).device


class WrapperModelFromHuggingFace(nn.Module):
    """
    Class wrapper for models from hugging face
    """

    def __init__(self, hugging_face_model: nn.Module):
        """
        Method for init model

        :param hugging_face_model: torch model from hugging face
        """
        super().__init__()
        self.model = hugging_face_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of model

        :param x: input tensor
        :return: model logits
        """
        return self.model(x).pooler_output
