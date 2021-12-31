from typing import Optional, Tuple, Union

import torch
import transformers

from torch import nn
from torchvision import models


class CLIP(nn.Module):
    """
    CLIP model with 2 parts:
        image part - CNN and text part - Transformer
    """

    def __init__(
            self,
            image_embedding: nn.Module,
            image_shape: int,
            text_embedding: nn.Module,
            text_shape: int
    ):
        """

        :param image_embedding: CNN for embedding image
        :param image_shape: dimension of image embedding
        :param text_embedding: Transform for embedding text
        :param text_shape: dimension of text embedding
        """
        super().__init__()
        # it's need for inference
        self.text_model = text_embedding
        # overall dim is 'text_shape'
        if image_shape == text_shape:
            self.image_model = image_embedding
        else:
            self.image_model = nn.Sequential(
                image_embedding,
                nn.Linear(in_features=image_shape, out_features=text_shape)
            )

    def _forward_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward method image part CLIP
        :param image: input image
        :return: normalized image embedding
        """
        image_embedding = self.image_model(image)
        image_features = image_embedding / image_embedding.norm(
            dim=-1,
            keepdim=True
        )
        return image_features

    def _forward_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward method text part CLIP
        :param text: input text description
        :return: normalized text embedding
        """
        text_embedding = self.text_model(text)
        text_features = text_embedding / text_embedding.norm(
            dim=-1,
            keepdim=True
        )
        return text_features

    def forward(
            self,
            image: torch.Tensor,
            text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method CLIP
        :param image: input image
        :param text: input text description
        :return: image and text logits
        """
        image_features = self._forward_image(image)
        text_features = self._forward_text(text)

        logits_image, logits_text = self._forward_cosine_similarity(
            image_features,
            text_features
        )

        return logits_image, logits_text

    @torch.no_grad()
    def inference(
            self,
            image: torch.Tensor,
            text: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference forward CLIP
        :param image: input image
        :param text: input text classes
        :return: classes of input images
        """
        logits_image, _ = self.forward(image, text)
        return torch.argmax(logits_image, dim=1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @staticmethod
    def _forward_cosine_similarity(
            image_features: torch.Tensor,
            text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function of cosine similarity of image and text vector embedding
        :param image_features: image embedding
        :param text_features: text embedding
        :return: tuple of image and text logits
        """
        logits_image = image_features @ text_features.t()
        logits_text = logits_image.t()
        return logits_image, logits_text


def configuration_image_model(
        name_model: str, *args, **kwargs
) -> Tuple[nn.Module, int]:
    """
    Function for init cnn model from torchvision.models
    :param name_model: name model from torchvision.models
    :param args: args for init model
    :param kwargs: kwargs for init model
    :return: cnn model and it's output vector dimension
    """
    if name_model in models.__dict__:
        try:
            # init model
            model = models.__dict__[name_model](*args, **kwargs)
            # change last layer and
            name_last_layer, last_layer = list(model.named_modules())[-1]
            output_shape = last_layer.in_features
            setattr(model, name_last_layer, nn.Identity())
            return model, output_shape
        except Exception as err:
            raise ValueError('Please type right image model name') from err
    else:
        raise ValueError('Please type right image model name')


class WrapperModelFromHuggingFace(nn.Module):

    def __init__(self, hugging_face_model: nn.Module):
        super().__init__()
        self.model = hugging_face_model

    def forward(self, x) -> torch.Tensor:
        return self.model(x)['logits']


def configuration_text_model(
        name_model: str, *args, **kwargs
) -> Tuple[nn.Module, int]:
    """
    Function for init transformer model from transformers (hugging face)
    :param name_model: name model from transformers
    :param args: args for init model
    :param kwargs: kwargs for init model
    :return: transformer and it's output vector dimension
    """
    if name_model in transformers.__dict__:
        try:
            if kwargs['pretrained'] and 'name_pretrained' in kwargs:
                model = transformers.__dict__[name_model].from_pretrained(
                    kwargs['name_pretrained']
                )
            else:
                model = transformers.__dict__[name_model](*args, **kwargs)
            name_last_layer, last_layer = list(model.named_modules())[-1]
            output_shape = last_layer.in_features
            setattr(model, name_last_layer, nn.Identity())
            return WrapperModelFromHuggingFace(model), output_shape
        except Exception as err:
            raise ValueError('Please type right text model name') from err
    else:
        raise ValueError('Please type right text model name')
