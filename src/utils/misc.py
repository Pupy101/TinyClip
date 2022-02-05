"""
Module with custom functions
"""

import os
import re

from typing import Optional

import torch

from torch import nn


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
