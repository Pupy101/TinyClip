import torch

from torch import nn


def freeze_weight(model: nn.Module):
    """
    Function for freeze weight in model
    :param model: torch model
    """
    for weight in model.parameters():
        weight.requires_grad = False


def create_label(index_of_pairs: torch.Tensor) -> torch.Tensor:
    size = len(index_of_pairs)
    label = torch.zeros((size, size))
    for i in range(size):
        label[i, :] = index_of_pairs == index_of_pairs[i]
    return label
