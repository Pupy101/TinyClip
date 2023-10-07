from math import ceil
from typing import List

from torch import nn


def freeze_model(model: nn.Module, train_part: float) -> None:
    assert 0 < train_part <= 1, f"train_part must be in (0, 1] (train_part: {train_part})"
    parameters = list(model.parameters())
    train_count = ceil(len(parameters) * train_part)
    for param in parameters[:-train_count]:
        param.requires_grad = False
