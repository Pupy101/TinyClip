"""
Module with custom functions
"""

import os
import re

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import requests
import torch

from torch import nn


def normalize(tensor: torch.Tensor) -> torch.tensor:
    """
    Function for normalize tensor along 1 dimension

    Args:
        tensor: input tensor

    Returns:
        normalized tensor
    """
    norm = torch.sqrt(torch.sum(tensor ** 2)).unsqueeze(1)
    return tensor / norm


def freeze_weight(model: nn.Module) -> None:
    """
    Function for freeze weight in model

    Args:
        model: torch model

    Returns:
        None
    """
    for weight in model.parameters():
        weight.requires_grad = False


def create_label(
        text_embedding: torch.Tensor,
        threshold: Optional[float] = 0.65
) -> torch.Tensor:
    """
    Function for creating labels from text embedding

    Args:
        text_embedding: normalized vector representation of text
        threshold: threshold for finding texts similarity by cosine similarity
            of vectors texts embeddings

    Returns:
        labels similarity of texts embedding
    """
    text_embedding = text_embedding.clone().detach()
    similarity = text_embedding @ text_embedding.t()
    return (similarity > threshold).float()


def find_max_predict_index(dir_to_predict, prefix_to_file) -> int:
    """
    Function for search last prediction and return it index

    Args:
        dir_to_predict: directory for predict
        prefix_to_file: prefix for file with predictions

    Returns:
        last index of file with predictions
    """
    pattern = f'{prefix_to_file}_([0-9]+)'
    files = os.listdir(dir_to_predict)
    max_file_index = 0
    for file in files:
        name, _ = os.path.splitext(file)
        match = re.match(pattern, name)
        if match:
            max_file_index = max(max_file_index, int(match.group(1)))
    return max_file_index + 1


def get_image(session: requests.Session, url: str, **kwargs):
    """
    Function for get image from url

    Args:
        session: request session
        url: path to image or url

    Returns:
        same string or bytes from url
    """
    resp = session.get(url=url, stream=True, **kwargs)
    if resp.status_code == 200:
        return resp.raw.read()
    return None


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


def compute_f1_batch(logits: torch.Tensor, target: torch.Tensor) -> Tuple[int, int, int]:
    """
    Function for compute TP, FP, FN from logits

    Args:
        logits: logits from CLIP
        target: target labels

    Returns:
        TP, FP, FN
    """
    predict = torch.round(logits)
    true_positive = predict * target
    false_positive = predict - true_positive
    false_negative = target - true_positive
    return true_positive.sum(), false_positive.sum(), false_negative.sum()
