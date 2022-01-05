import torch

from torch import nn


def freeze_weight(model: nn.Module):
    """
    Function for freeze weight in model
    :param model: torch model
    """
    for weight in model.parameters():
        weight.requires_grad = False


def create_label_from_index(index_of_pairs: torch.Tensor) -> torch.Tensor:
    size = len(index_of_pairs)
    label = torch.zeros((size, size))
    for i in range(size):
        label[i, :] = index_of_pairs == index_of_pairs[i]
    return label


def create_label_from_text(text_vectors: torch.Tensor) -> torch.Tensor:
    text_vectors = text_vectors.clone().detach()  # batch size x vector dim
    similarity = text_vectors @ text_vectors.t()
    return (similarity > 0.5).float()
