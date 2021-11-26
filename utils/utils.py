from typing import List

from torch import nn


def weights_change_requires_grad(
        freeze_layers: List[nn.Parameter] = None,
        unfreeze_layers: List[nn.Parameter] = None
):
    if freeze_layers is not None:
        for weight in freeze_layers:
            weight.requires_grad = False
    if unfreeze_layers is not None:
        for weight in unfreeze_layers:
            weight.requires_grad = True
