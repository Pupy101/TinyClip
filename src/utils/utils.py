"""
Module with custom functions
"""

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
        threshold: Optional[float] = 0.5
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
