"""
This module consist some wrappers for simple using text models from hugging face
and class for creation vision model for CLIP from pretrained models torchvision
"""

from typing import List, Optional, Type

import torch

from torch import nn
from torchvision import models

from ..utils.functions import get_annotation_from_parent_model


class WrapperModelFromHuggingFace(nn.Module):
    """
    Class wrapper for models from hugging face
    """

    def __init__(
            self,
            hugging_face_model: nn.Module,
            getting_attr_from_model: Optional[str] = None,
    ):
        """
        Method for init wrapped model

        Args:
            hugging_face_model: torch model from hugging face
        """
        super().__init__()
        self.model = hugging_face_model
        self.get_attr = getting_attr_from_model
        get_annotation_from_parent_model(self, hugging_face_model)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward method of model

        Args:
            *args: args for model
            **kwargs: kwargs for model
        Returns:
            embedding of input text description
        """
        output = self.model(*args, **kwargs)
        if self.get_attr is not None:
            return output.__dict__[self.get_attr]
        return output


class VisionModelPreparator:
    def __init__(self, model: str, *args, **kwargs):
        """
        Init method for Vision model

        Args:
            model: vision model from torchvision.models
            *args: args for initialization model
            **kwargs: kwargs for initialization model
        """
        model_name = model
        self.cv = models.__dict__[model_name](*args, **kwargs)

    def change_layer_to_mlp(
            self,
            layer_name: str,
            mlp_shapes: List[int],
            activation: Type[nn.Module],
    ) -> 'VisionModelPreparator':
        """
        This method change last part of model - classifier

        Args:
            layer_name: name of changing layer in network
            mlp_shapes: shape [input,  ..., hidden, ..., output] shape in mlp
            activation: activation function between linear layers

        Returns:
            model with replaced classifier part
        """
        mlp = self.create_mlp(
            shapes=mlp_shapes, activation=activation
        )
        setattr(self.cv, layer_name, mlp)
        return self

    @staticmethod
    def create_mlp(
            shapes: List[int], activation: Type[nn.Module]
    ) -> nn.Module:
        """
        Method for create Linear net with activation between layers

        Args:
            shapes: size of input and output features from layers
            activation: activation function between linear layers

        Returns:
            created mlp with activation function
        """
        classifier = []
        for i in range(len(shapes) - 1):
            classifier.append(nn.Linear(shapes[i], shapes[i + 1]))
            if i < i - 2:
                # adding activation only between Linear layers
                classifier.append(activation())
        # at the end append Tanh activation
        classifier.append(nn.Tanh())
        return nn.Sequential(*classifier)

    @property
    def model(self):
        return self.cv

