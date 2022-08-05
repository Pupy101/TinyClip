from typing import List, Optional, Tuple, Union

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
        count_accumulated_batches: int = 1,
    ) -> None:
        self.clip = clip
        self.criterion_clip = criterion_clip
        self.criterion_image = criterion_image
        self.criterion_text = criterion_text
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.acc_count = count_accumulated_batches

        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()

        self.clip_image_outputs: List[Tensor] = []
        self.clip_text_outputs: List[Tensor] = []
        self.clip_labels: List[Tensor] = []
        self.image_outputs: List[Tensor] = []
        self.image_labels: List[Tensor] = []
        self.text_outputs: List[Tensor] = []
        self.text_labels: List[Tensor] = []
        self.counter_clip_steps = 0
        self.counter_image_steps = 0
        self.counter_text_steps = 0

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
        self.counter_clip_steps = 0
        self.counter_image_steps = 0
        self.counter_text_steps = 0

    def eval(self) -> None:
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.eval()
        self.counter_clip_steps = 0
        self.counter_image_steps = 0
        self.counter_text_steps = 0

    def batch_on_device(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return tuple(batch_element.to(self.device) for batch_element in batch)

    def clip_forward(self, batch: Tuple[Tensor, Tensor]) -> Optional[Tensor]:
        self.counter_clip_steps += 1

        image, text = self.batch_on_device(batch)
        output = self.clip.forward(image=image, text=text)
        labels = torch.diag(torch.ones(image.size(0))).to(self.device)

        self.clip_image_outputs.append(output.embeddings.image)
        self.clip_text_outputs.append(output.embeddings.text)
        self.clip_labels.append(labels)

        if self.counter_clip_steps % self.acc_count:
            return None

        image_embeddings = torch.cat(self.clip_image_outputs, dim=0)
        text_embeddings = torch.cat(self.clip_text_outputs, dim=0)
        acc_labels = torch.cat(self.clip_labels, dim=0)
        logit_scale = self.clip.logit_scale.exp()
        logits_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_text = logits_image.t()

        loss_img: Tensor = self.criterion_clip(logits_image, acc_labels)
        loss_text: Tensor = self.criterion_clip(logits_text, acc_labels)
        loss = loss_img + loss_text
        tp, fp, fn = compute_f1_batch(logits_image, acc_labels)
        self.clip_metrics.update(
            tp=tp, fp=fp, fn=fn, loss=loss.item(), count=image.size(0)
        )
        return loss

    def image_part_forward(self, batch: Tuple[Tensor, Tensor]) -> Optional[Tensor]:
        self.counter_image_steps += 1

        image, label = self.batch_on_device(batch)
        logits = self.clip.vision_part.forward(image=image, is_classification=True)
        self.image_outputs.append(logits)
        self.image_outputs.append(label)

        if self.counter_image_steps % self.acc_count:
            return None

        image_logits = torch.cat(self.image_outputs, dim=0)
        image_labels = torch.cat(self.image_labels, dim=0)

        loss: Tensor = self.criterion_image(image_logits, image_labels)
        top1 = compute_accuracy_1(image_logits, image_labels)
        top5 = compute_accuracy_5(image_logits, image_labels)
        count = image_logits.size(0)
        self.image_metrics.update(top1=top1, top5=top5, loss=loss.item(), count=count)
        return loss

    def text_part_forward(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Optional[Tensor]:
        self.counter_text_steps += 1

        masked_ids, perm_mask, label_ids = self.batch_on_device(batch)
        logits = self.clip.text_part.forward(
            text=masked_ids, perm_mask=perm_mask, is_mlm=True
        )
        self.text_outputs.append(logits)
        self.text_labels.append(label_ids)

        if self.counter_image_steps % self.acc_count:
            return None

        text_logits = torch.cat(self.text_outputs, dim=0)
        text_labels = torch.cat(self.text_labels, dim=0)

        loss: Tensor = self.criterion_text(text_logits, text_labels)
        top1 = compute_accuracy_1(text_logits.permute(0, 2, 1), text_labels)
        top5 = compute_accuracy_5(text_logits.permute(0, 2, 1), text_labels)
        count = text_logits.size(0) * text_logits.size(1)
        self.text_metrics.update(top1=top1, top5=top5, loss=loss.item(), count=count)
        return loss

    def optimization_step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
