import torch

from torch import nn


def freeze_weight(model: nn.Module) -> None:
    """
    Function for freeze weight in model

    :param model: torch model
    """
    for weight in model.parameters():
        weight.requires_grad = False


def create_label_from_text_embedding(
        text_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Function for creating labels from text embedding

    :param text_embedding: normalized vector representation of text
    from text model
    :return: labels similarity of texts embedding
    """
    text_embedding = text_embedding.clone().detach()
    similarity = text_embedding @ text_embedding.t()
    return (similarity > 0.5).float()
