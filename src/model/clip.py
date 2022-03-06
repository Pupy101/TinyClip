"""
Module with CLIP model. It contains separated vision part of CLIP.
Vision part is separated, because only it will be trained in this library.

In module text/image features is output from vision and text model.
And text/image embeddings is normalized text/image features.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from torch import nn

from src.utils.functions import normalize
from src.classes.model import CLIPInferenceOutput, CLIPOutput, Embeddings, Logits


class VisionPartCLIP(nn.Module):
    """
    Vision part of clip with computing CLIP logits
    """
    def __init__(self, model: nn.Module):
        """
        Method for init vision part CLIP

        Args:
            model: CNN model for getting image embedding
        """
        super().__init__()
        self.cv_model = model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
            self,
            image: torch.Tensor,
            text_embedding: torch.Tensor,
            image_features: Optional[torch.Tensor] = None,
            only_features: Optional[bool] = False,
    ) -> CLIPOutput:
        """
        Forward method of vision part with computing image and text embedding
        cosine similarity

        Args:
            image: input image
            text_embedding: text embedding from text model
            image_features: vector representation of image from vision model
            only_features: return only image and text features without logits

        Returns:
            image logits, text logits, image embedding
        """
        if image_features is None:
            image_features = self.cv_model(image)
        image_embedding = normalize(image_features)
        if only_features:
            return CLIPOutput(
                embeddings=Embeddings(image=image_embedding, text=text_embedding),
                logits=Logits(),
            )
        logits = self.compute_logit(
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            logit_scale=self.logit_scale,
        )
        return CLIPOutput(
            embeddings=Embeddings(image=image_embedding, text=text_embedding),
            logits=Logits(image=logits.image, text=logits.text),
        )

    @staticmethod
    def compute_logit(
            image_embedding: torch.Tensor, text_embedding: torch.Tensor, logit_scale: nn.Parameter
    ) -> Logits:
        """
        Method for compute logits between image and text embeddings

        Args:
            image_embedding: image embedding
            text_embedding: text embedding
            logit_scale: scale for transform cosine similarity into logits

        Returns:
            image logits, text logits
        """
        image_logit = logit_scale * image_embedding @ text_embedding.t()
        text_logit = image_logit.t()
        return Logits(image=image_logit, text=text_logit)

    @property
    def device(self) -> torch.device:
        """
        Attribute with device on which the model is located.

        Returns:
            model device
        """
        return next(iter(self.parameters())).device


class CLIP(nn.Module):
    """
    CLIP model with 2 parts:
    image part and text part
    """
    def __init__(self, vision_part: VisionPartCLIP, text_model: nn.Module):
        """
        Method for init CLIP

        Args:
            vision_part: vision part of clip
            text_model: text model for getting text features
        """
        super().__init__()
        self.cv_model = vision_part
        self.text_model = text_model

    def forward(
            self,
            image: torch.Tensor,
            text: torch.Tensor,
            image_features: Optional[torch.Tensor] = None,
            text_features: Optional[torch.Tensor] = None,
            only_features: Optional[bool] = False
    ) -> CLIPOutput:
        """
        Forward method CLIP

        Args:
            image: input image
            text: text description
            image_features: vector representation of image from vision model
            text_features: vector representation of text from text model
            only_features: return only features or features with image and
                text logits

        Returns:
            image and text logits, (image and text embeddings)
        """
        if text_features is None:
            text_features = self.text_model(text)
        text_embedding = normalize(text_features)
        return self.cv_model(
            image=image,
            text_embedding=text_embedding,
            image_features=image_features,
            only_features=only_features,
        )

    @torch.no_grad()
    def inference(
            self,
            image: torch.Tensor,
            text: torch.Tensor,
            image_features: Optional[torch.Tensor] = None,
            text_features: Optional[torch.Tensor] = None,
            raw_output: Optional[bool] = False
    ) -> CLIPInferenceOutput:
        """
        Inference forward CLIP

        Args:
            image: input image
            text: input text classes
            image_features: images embedding vectors
            text_features: text embedding vectors
            raw_output: is raw output from model

        Returns:
            classes of input images
        """
        output: CLIPOutput = self.forward(image, text, image_features, text_features)
        classes = torch.argmax(output.logits.image, dim=1)
        if raw_output:
            return CLIPInferenceOutput(
                classes=classes,
                embeddings=output.embeddings
            )
        return CLIPInferenceOutput(classes=classes, embeddings=Embeddings())

    @property
    def device(self) -> torch.device:
        """
        Method for know device on which the model is located.

        Returns:
            model device
        """
        return next(iter(self.parameters())).device
