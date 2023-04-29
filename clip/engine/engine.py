from typing import Dict, List, Optional, Tuple, overload

import torch
from blocks.utils import set_seed
from torch import Tensor, optim
from transformers import BatchEncoding

from clip.engine.metrics import ClassificationMetrics, CLIPMetrics, MaskedLMMetric
from clip.models import CLIP
from clip.types import Device, MultiTaskCriterions, MultiTaskProportions, Scheduler
from clip.utils.metrics import compute_accuracy1, compute_accuracy5, compute_f1


class Engine:
    def __init__(
        self,
        clip: CLIP,
        criterions: MultiTaskCriterions,
        coefficients: MultiTaskProportions,
        optimizer: optim.Optimizer,
        device: Device,
        scheduler: Optional[Scheduler] = None,
        count_accumulating_steps: int = 1,
        seed: int = 0xFEED,
    ) -> None:
        self.clip = clip
        self.criterions = criterions
        self.coefficients = coefficients
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.acc_count = count_accumulating_steps

        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ClassificationMetrics()
        self.text_metrics = MaskedLMMetric()

        self.clip_image_outputs: List[Tensor] = []
        self.clip_text_outputs: List[Tensor] = []
        self.counter_clip_steps = 0

        self.clip.to(self.device)

        set_seed(seed)

    def train(self) -> None:
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.train()
        self.optimizer.zero_grad()
        self.reset_batch_accumulation_clip()

    def eval(self) -> None:
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.eval()
        self.reset_batch_accumulation_clip()

    def reset_batch_accumulation_clip(self) -> None:
        self.clip_image_outputs = []
        self.clip_text_outputs = []
        self.counter_clip_steps = 0

    @overload
    def batch_on_device(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        ...

    @overload
    def batch_on_device(self, batch: List[Tensor]) -> List[Tensor]:
        ...

    @overload
    def batch_on_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ...

    @overload
    def batch_on_device(self, batch: BatchEncoding) -> BatchEncoding:
        ...

    def batch_on_device(self, batch):
        if isinstance(batch, tuple):
            return tuple(batch_element.to(self.device) for batch_element in batch)
        if isinstance(batch, list):
            return [batch_element.to(self.device) for batch_element in batch]
        if isinstance(batch, BatchEncoding):
            batch.to(self.device)
            return batch
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        raise ValueError(f"Strange value for batch_on_device: {batch}")

    def clip_forward(self, batch: BatchEncoding) -> Optional[Tensor]:
        self.counter_clip_steps += 1

        batch = self.batch_on_device(batch)

        image = batch.pop("image")
        text = batch

        image_features = self.clip.image_part.forward(image)
        text_features = self.clip.text_part.forward(**text)  # pylint: disable=not-a-mapping

        output = self.clip.forward(image_features=image_features, text_features=text_features)

        self.clip_image_outputs.append(output.embeddings.image)
        self.clip_text_outputs.append(output.embeddings.text)

        if self.counter_clip_steps % self.acc_count:
            return None

        image_embeddings = torch.cat(self.clip_image_outputs, dim=0)
        text_embeddings = torch.cat(self.clip_text_outputs, dim=0)
        logits = self.clip.compute_logit(image=image_embeddings, text=text_embeddings)

        labels = torch.diag(torch.ones(logits.image.size(0))).to(self.device)
        loss: Tensor
        loss = self.criterions.clip(logits.image, labels) + self.criterions.clip(logits.text, labels)
        self.clip_metrics.update(loss=loss.item(), count=image_embeddings.size(0), **compute_f1(logits.image, labels))
        self.reset_batch_accumulation_clip()
        return loss * self.coefficients.clip

    def image_part_forward(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = self.batch_on_device(batch)  # pylint: disable=unpacking-non-sequence
        label = batch["label"]
        logits = self.clip.image_part.forward(image=batch["image"], classification=True)

        loss: Tensor = self.criterions.image(logits, label)
        self.image_metrics.update(
            top1=compute_accuracy1(logits, label),
            top5=compute_accuracy5(logits, label),
            loss=loss.item(),
            count=logits.size(0),
        )
        return loss * self.coefficients.image

    def text_part_forward(self, batch: BatchEncoding) -> Tensor:
        batch = self.batch_on_device(batch)
        labels = batch.pop("labels")
        logits = self.clip.text_part.forward(**batch, masked_lm=True)  # pylint: disable=not-a-mapping

        loss: Tensor = self.criterions.text(logits, labels)
        self.text_metrics.update(
            top1=compute_accuracy1(logits.permute(0, 2, 1), labels),
            top5=compute_accuracy5(logits.permute(0, 2, 1), labels),
            loss=loss.item(),
            count=logits.size(0),
        )
        return loss * self.coefficients.text

    def optimization_step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
