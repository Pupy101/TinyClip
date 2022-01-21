"""
This module consist some wraps for simple using text models from hugging face
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch

from torch import nn


class WrapperModelFromHuggingFace(nn.Module):
    """
    Class wrapper for models from hugging face
    """

    def __init__(self, hugging_face_model: nn.Module):
        """
        Method for init wrapped model

        Args:
            hugging_face_model: torch model from hugging face
        """
        super().__init__()
        self.model = hugging_face_model
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
        return self.model(*args, **kwargs).pooler_output


def get_annotation_from_parent_model(
        child_model: nn.Module,
        parent_model: nn.Module,
        depth_recursion: int = 1
) -> None:
    """
    Function for getting describing attributes from parent model

    Args:
        child_model: child or wrapped model
        parent_model: parent model
        depth_recursion: depth recursion

    Returns:
        None
    """
    method_from_parent: Set[str] = {
        '__annotations__', '__doc__', '__module__', '__name__', '__qualname__'
        }
    methods_and_attrs: Dict[int, List[Tuple[object, object]]] = defaultdict(list)
    methods_and_attrs[0].append((parent_model, child_model))
    for depth in range(depth_recursion):
        parent_and_child_attributes = methods_and_attrs[depth]
        for parent, child in parent_and_child_attributes:
            parent_attrs, child_attrs = set(dir(parent)), set(dir(child))
            intersection_attrs = parent_attrs & child_attrs
            changed_attrs = intersection_attrs & method_from_parent
            for updated_attr in changed_attrs:
                try:
                    setattr(child, updated_attr, getattr(parent, updated_attr))
                except AttributeError:
                    print(
                        f'Can\'t set attribute{updated_attr} from'
                        f' {parent.__name__} to {child.__name__}'
                    )
            methods_and_attrs[depth + 1].extend([
                (getattr(parent, attr), getattr(child, attr))
                for attr in intersection_attrs if not attr.startswith('_')
            ])
