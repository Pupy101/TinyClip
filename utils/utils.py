from torch import nn


def freeze_weights(model: nn.Module, last_index_freeze: int):
    for weight in list(model.parameters())[:last_index_freeze]:
        weight.requires_grad = False


def unfreeze_weights(model: nn.Module):
    for weight in model.parameters():
        weight.requires_grad = True
