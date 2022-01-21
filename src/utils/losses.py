"""
Module with custom losses
"""

import torch
import torch.nn.functional as F

from torch import nn


class FocalLoss(nn.Module):
    """
    Focal loss from https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = 'mean'
    ) -> None:
        """
        Method for init dataset

        Args:
            alpha: coefficient of loss
            gamma: coefficient of loss
            reduction: type of reduction None|'sum'|'mean'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets) -> torch.Tensor:
        """
        Method for computing loss

        Args:
            inputs: logits from model
            targets: targets labels

        Returns:
            final value of computed loss
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
