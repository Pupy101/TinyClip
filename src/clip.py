from typing import Tuple, Union

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
        Initialization model
            :param image_embedding: name of image embedding net
            :param text_embedding: name of text embedding net
        """
        super().__init__()
        # it's need for inference
        self.__computed_text_embedding = None
        self.model_image_embedding = image_embedding
        self.model_text_embedding = text_embedding
        # overall dim is 'text_shape'
        if image_shape == text_shape:
            self.linear_image = nn.Identity()
        else:
            self.linear_image = nn.Linear(
                in_features=image_shape,
                out_features=text_shape
            )

    def forward_image_part(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward method image part CLIP
        :param img: input image
        :return: normalized image embedding
        """
        image_embedding = self.model_image_embedding(img)
        reshaped_img_emb = self.linear_image(image_embedding)
        image_features = reshaped_img_emb / reshaped_img_emb.norm(
            dim=-1,
            keepdim=True
        )
        return image_features

    def forward_text_part(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward method text part CLIP
        :param text: input text description
        :return: normalized text embedding
        """
        text_embedding = self.model_text_embedding(text)
        text_features = text_embedding / text_embedding.norm(
            dim=-1,
            keepdim=True
        )
        return text_features

    def forward(
            self,
            vectors: Tuple[torch.Tensor, torch.Tensor],
            image_features: torch.Tensor = None,
            text_features: torch.Tensor = None,
            is_raw_output: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Forward method CLIP model
        :param vectors: tuple tensors of image and text
        :param image_features: image embedding
        :param text_features: text embedding
        :param is_raw_output: is return all output (logits and embeddings
        of image and text)
        :return: logits or logits and embeddings
        """
        img, text = vectors

        if image_features is None:
            image_features = self.forward_image_part(img)

        if text_features is None:
            text_features = self.forward_text_part(text)

        (
            logits_image,
            logits_text
        ) = forward_cosine_similarity(
            image_features,
            text_features
        )

        if is_raw_output:
            return logits_image, logits_text, image_features, text_features

        return logits_image, logits_text

    @torch.no_grad()
    def inference(
            self,
            input_tensor: Tuple[torch.Tensor, torch.Tensor],
            is_rewrite_classes: bool = False
    ) -> torch.Tensor:
        """
        Method for inference CLIP
        :param input_tensor: tuple of images and text token as torch.tensor
        :param is_rewrite_classes: model cashing classes from text and ignore text input
            after one inference. Classes from text will updated If this flag is True
        :return: classes for images
        """
        image, text = input_tensor
        if (
                self.__computed_text_embedding is not None
                and not is_rewrite_classes
        ):
            computed_text_embedding = self.__computed_text_embedding
            logits_image, _ = self.forward(
                image, text, text_features=computed_text_embedding
            )
        else:
            logits_image, *_, text_features = self.forward(image, text, is_raw_output=True)
            self.__computed_text_embedding = text_features
        return torch.argmax(logits_image, dim=1)

    @property
    def device(self):
        return next(iter(self.parameters())).device


def forward_cosine_similarity(
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
        name_model: str,
        *args,
        **kwargs
) -> Tuple[nn.Module, int]:
    """
    Function for init cnn model from torchvision.models
    :param name_model: name model from torchvision.models
    :param args: args for init model
    :param kwargs: kwargs for init model
    :return: cnn model
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


def configuration_text_model(name_model: str, *args, **kwargs) -> Tuple[nn.Module, int]:
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
            return model, output_shape
        except Exception as err:
            raise ValueError('Please type right text model name') from err
    else:
        raise ValueError('Please type right text model name')
