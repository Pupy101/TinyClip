from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn, optim

from src.models import CLIP
from src.utils.functions import compute_accuracy_1, compute_accuracy_5, compute_f1_batch

from .metrics import CLIPMetrics, ImageClassificationMetrics, MaskedLMMetric


class Engine:
    def __init__(
        self,
        seed: int,
        clip: CLIP,
        criterion_clip: nn.Module,
        criterion_image: nn.Module,
        criterion_text: nn.Module,
        optimizer: optim.Optimizer,
        device: Union[str, torch.device],
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.clip = clip
        self.criterion_clip = criterion_clip
        self.criterion_image = criterion_image
        self.criterion_text = criterion_text
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.to(self.device)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def train(self) -> None:
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.train()
        self.optimizer.zero_grad()

    def eval(self) -> None:
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.eval()

    def batch_on_device(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return tuple(batch_element.to(self.device) for batch_element in batch)

    def clip_forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        image, text = self.batch_on_device(batch)
        output = self.clip.forward(image=image, text=text)
        labels = torch.diag(torch.ones(image.size(0))).to(self.device)
        loss: Tensor = self.criterion_clip(
            output.logits.image, labels
        ) + self.criterion_clip(output.logits.text, labels)
        tp, fp, fn = compute_f1_batch(output.logits.image, labels)
        count = image.size(0)
        self.clip_metrics.update(tp=tp, fp=fp, fn=fn, loss=loss.item(), count=count)
        return loss

    def image_part_forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        image, label = self.batch_on_device(batch)
        logits = self.clip.vision_part.forward(image=image, is_classification=True)
        loss: Tensor = self.criterion_image(logits, label)
        top1 = compute_accuracy_1(logits, label)
        top5 = compute_accuracy_5(logits, label)
        count = image.size(0)
        self.image_metrics.update(top1=top1, top5=top5, loss=loss.item(), count=count)
        return loss

    def text_part_forward(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        masked_ids, perm_mask, label_ids = self.batch_on_device(batch)
        logits = self.clip.text_part.forward(
            text=masked_ids, perm_mask=perm_mask, is_mlm=True
        )
        loss: Tensor = self.criterion_text(logits, label_ids)
        top1 = compute_accuracy_1(logits.permute(0, 2, 1), label_ids)
        top5 = compute_accuracy_5(logits.permute(0, 2, 1), label_ids)
        count = masked_ids.size(0) * masked_ids.size(1)
        self.text_metrics.update(top1=top1, top5=top5, loss=loss.item(), count=count)
        return loss

    def optimization_step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
