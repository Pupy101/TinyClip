from typing import Dict, List, Optional, Tuple, TypeVar, overload

import numpy as np
import torch
from torch import Tensor, optim
from transformers import BatchEncoding

from clip.models import CLIP
from clip.types import Device, MultiTaskCriterions, Scheduler
from clip.utils.metrics import compute_accuracy1, compute_accuracy5, compute_f1

from .metrics import CLIPMetrics, ImageClassificationMetrics, MaskedLMMetric

Batch = TypeVar("Batch", Dict[str, Tensor], BatchEncoding)


class Engine:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        clip: CLIP,
        criterion: MultiTaskCriterions,
        optimizer: optim.Optimizer,
        device: Device,
        scheduler: Optional[Scheduler] = None,
        count_accumukating_steps: int = 1,
        seed: int = 0xFEED,
    ) -> None:
        self.clip = clip
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.acc_count = count_accumukating_steps

        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()

        self.clip_image_outputs: List[Tensor] = []
        self.clip_text_outputs: List[Tensor] = []
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
        self._reset_batch_accumulation()

    def eval(self) -> None:
        self.clip_metrics = CLIPMetrics()
        self.image_metrics = ImageClassificationMetrics()
        self.text_metrics = MaskedLMMetric()
        self.clip.eval()
        self._reset_batch_accumulation()

    def _reset_batch_accumulation(self) -> None:
        self.clip_image_outputs = []
        self.clip_text_outputs = []
        self.image_outputs = []
        self.image_labels = []
        self.text_outputs = []
        self.text_labels = []
        self.counter_clip_steps = 0
        self.counter_image_steps = 0
        self.counter_text_steps = 0

    @overload
    def batch_on_device(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        ...

    @overload
    def batch_on_device(self, batch: Batch) -> Batch:
        ...

    def batch_on_device(self, batch):
        if isinstance(batch, tuple):
            return tuple(batch_element.to(self.device) for batch_element in batch)
        if isinstance(batch, BatchEncoding):
            batch.to(self.device)
            return batch
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        raise ValueError("Strange value for batch_on_device: {batch}")

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
        loss = self.criterion.clip(logits.image, labels) + self.criterion.clip(logits.text, labels)
        self.clip_metrics.update(
            loss=loss.item(), count=image_embeddings.size(0), **compute_f1(logits.image, labels)
        )
        self._reset_batch_accumulation()
        return loss

    def image_part_forward(self, batch: Tuple[Tensor, Tensor]) -> Optional[Tensor]:
        self.counter_image_steps += 1

        image, label = self.batch_on_device(batch)  # pylint: disable=unpacking-non-sequence
        logits = self.clip.image_part.forward(image=image, classification=True)
        self.image_outputs.append(logits)
        self.image_labels.append(label)

        if self.counter_image_steps % self.acc_count:
            return None

        image_logits = torch.cat(self.image_outputs, dim=0)
        image_labels = torch.cat(self.image_labels, dim=0)

        loss: Tensor = self.criterion.image(image_logits, image_labels)
        self.image_metrics.update(
            top1=compute_accuracy1(image_logits, image_labels),
            top5=compute_accuracy5(image_logits, image_labels),
            loss=loss.item(),
            count=image_logits.size(0),
        )
        self._reset_batch_accumulation()
        return loss

    def text_part_forward(self, batch: BatchEncoding) -> Optional[Tensor]:
        self.counter_text_steps += 1

        batch = self.batch_on_device(batch)
        labels = batch.pop("labels")
        logits = self.clip.text_part.forward(
            **batch, masked_lm=True  # pylint: disable=not-a-mapping
        )
        self.text_outputs.append(logits)
        self.text_labels.append(labels)

        if self.counter_text_steps % self.acc_count:
            return None

        text_logits = torch.cat(self.text_outputs, dim=0)
        text_labels = torch.cat(self.text_labels, dim=0)

        loss: Tensor = self.criterion.text(text_logits, text_labels)
        self.text_metrics.update(
            top1=compute_accuracy1(text_logits.permute(0, 2, 1), text_labels),
            top5=compute_accuracy5(text_logits.permute(0, 2, 1), text_labels),
            loss=loss.item(),
            count=text_logits.size(0),
        )
        self._reset_batch_accumulation()
        return loss

    def optimization_step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
