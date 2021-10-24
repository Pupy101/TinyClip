from torch import nn


def freeze_weights(model: nn.Module, last_index_freeze: int, freeze_all_net: bool = False):
    if freeze_all_net:
        weights = model.parameters()
    else:
        weights = list(model.parameters())[:last_index_freeze]
    for weight in weights:
        weight.requires_grad = False


def unfreeze_weights(model: nn.Module):
    for weight in model.parameters():
        weight.requires_grad = True
