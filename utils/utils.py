from torch import nn


def freeze_weight(model: nn.Module):
    for weight in model.parameters():
        weight.requires_grad = False
