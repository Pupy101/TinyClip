"""
Module with custom functions
"""

from typing import Iterator, Optional, Tuple, TypeVar

import torch
from torch import Tensor, nn

Batch = TypeVar("Batch")


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Function for normalize tensor along 1 dimension."""
    norm = torch.sqrt(torch.sum(tensor**2, dim=1)).unsqueeze(1)
    return tensor / norm


def freeze_weight(model: nn.Module) -> None:
    """Function for freeze weight in model."""
    for weight in model.parameters():
        weight.requires_grad = False


def compute_f1_batch(logits: Tensor, target: Tensor) -> Tuple[float, float, float]:
    """Function for compute TP, FP, FN from logits."""
    predict = torch.round(torch.softmax(logits.detach(), dim=-1))
    true_positive = predict * target
    false_positive = predict - true_positive
    false_negative = target - true_positive
    return (
        true_positive.sum().cpu().item(),
        false_positive.sum().cpu().item(),
        false_negative.sum().cpu().item(),
    )


def compute_accuracy_1(logits: Tensor, target: Tensor) -> int:
    """Function for compute right classificated classes."""
    output_labels = torch.argmax(logits.detach(), dim=-1)
    return round(torch.sum(output_labels == target).cpu().item())


def compute_accuracy_5(logits: Tensor, target: Tensor) -> int:
    """Function for compute right classificated top5 classes."""
    output_labels = torch.topk(logits.detach(), k=5, dim=-1).indices
    target = target.unsqueeze(-1)
    return round(torch.sum(output_labels == target).cpu().item())


def get_batch(loader: Iterator[Batch]) -> Optional[Batch]:
    try:
        batch = next(loader)
        return batch
    except StopIteration:
        return None
