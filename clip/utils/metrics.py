from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def freeze_weight(model: nn.Module) -> None:
    for weight in model.parameters():
        weight.requires_grad = False


def compute_f1(logits: Tensor, target: Tensor) -> Dict[str, float]:
    predict = torch.softmax(logits.detach(), dim=-1)
    tp = predict * target
    fp = predict - tp
    fn = target - tp
    tp, fp, fn = tp.sum().cpu(), fp.sum().cpu(), fn.sum().cpu()
    return {"tp": tp.item(), "fp": fp.item(), "fn": fn.item()}


def compute_accuracy1(logits: Tensor, target: Tensor) -> int:
    output_labels = torch.argmax(logits.detach(), dim=-1)
    return round(torch.sum(output_labels == target).cpu().item())


def compute_accuracy5(logits: Tensor, target: Tensor) -> int:
    output_labels = torch.topk(logits.detach(), k=5, dim=-1).indices
    target = target.unsqueeze(-1)
    return round(torch.sum(output_labels == target).cpu().item())
