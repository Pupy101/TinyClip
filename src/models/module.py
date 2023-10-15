from typing import Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, arange, argmax, nn
from torch.nn import functional as F
from torchmetrics import MeanMetric, Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import BatchEncoding

from src.models.clip import Clip
from src.types import Optimizer, Scheduler
from src.utils import freeze_model


class CLIPModule(LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_train_part: float,
        text_encoder: nn.Module,
        text_train_part: float,
        optimizer: Optimizer,  # pylint: disable=unused-argument
        scheduler: Optional[Scheduler],  # pylint: disable=unused-argument
        label_smoothing: Optional[float],
        batch_size: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters("optimizer", "scheduler", logger=False)

        freeze_model(model=image_encoder, train_part=image_train_part)
        freeze_model(model=text_encoder, train_part=text_train_part)

        self.clip = Clip(image_encoder=image_encoder, text_encoder=text_encoder)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing or 0.0)

        self.loss_train = MeanMetric()
        self.loss_val = MeanMetric()
        self.loss_test = MeanMetric()

        self.acc_train = MulticlassAccuracy(num_classes=batch_size, average="weighted")
        self.acc_val = MulticlassAccuracy(num_classes=batch_size, average="weighted")
        self.acc_test = MulticlassAccuracy(num_classes=batch_size, average="weighted")

        self.f1_train = MulticlassF1Score(num_classes=batch_size, average="weighted")
        self.f1_val = MulticlassF1Score(num_classes=batch_size, average="weighted")
        self.f1_test = MulticlassF1Score(num_classes=batch_size, average="weighted")

    def forward(self, batch: BatchEncoding) -> Tuple[Tensor, Tensor]:
        images = batch.pop("images")
        image_embeddings = self.clip.normalize(self.clip.image_encoder(images).logits)
        text_embeddings = self.clip.normalize(self.clip.text_encoder(**batch).logits)
        image_logits, text_logits = self.clip(image_embeddings=image_embeddings, text_embeddings=text_embeddings)
        return image_logits, text_logits

    def model_step(self, batch: BatchEncoding) -> Tuple[Tensor, Tensor, Tensor]:
        image_logits, text_logits = self.forward(batch=batch)
        labels = arange(image_logits.size(0), device=self.device)
        loss = self.criterion(image_logits, labels) + self.criterion(text_logits, labels)
        predict = argmax(image_logits, dim=-1)
        return loss, predict, labels

    def on_train_start(self) -> None:
        self.loss_val.reset()
        self.acc_val.reset()
        self.f1_val.reset()

    def compute_metric(
        self,
        prefix: str,
        loss: Tensor,
        predict: Tensor,
        labels: Tensor,
        loss_metric: Metric,
        acc_metric: Metric,
        f1_metric: Metric,
    ) -> None:
        loss_metric(loss)
        acc = acc_metric(predict, labels)
        f1 = f1_metric(predict, labels)

        self.log(prefix + "/loss", self.loss_train, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(prefix + "/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(prefix + "/f1", f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch: BatchEncoding, _: int) -> Tensor:
        loss, predict, labels = self.model_step(batch)

        self.compute_metric(
            prefix="train",
            loss=loss,
            predict=predict,
            labels=labels,
            loss_metric=self.loss_train,
            acc_metric=self.acc_train,
            f1_metric=self.f1_train,
        )

        return loss

    def validation_step(self, batch: BatchEncoding, _: int) -> None:
        loss, predict, labels = self.model_step(batch)

        self.compute_metric(
            prefix="val",
            loss=loss,
            predict=predict,
            labels=labels,
            loss_metric=self.loss_val,
            acc_metric=self.acc_val,
            f1_metric=self.f1_val,
        )

    def test_step(self, batch: BatchEncoding, _: int) -> None:
        loss, predict, labels = self.model_step(batch)

        self.compute_metric(
            prefix="test",
            loss=loss,
            predict=predict,
            labels=labels,
            loss_metric=self.loss_test,
            acc_metric=self.acc_test,
            f1_metric=self.f1_test,
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())  # type: ignore
        if self.hparams.scheduler is not None:  # type: ignore[attr-defined]
            scheduler = self.hparams.scheduler(optimizer=optimizer)  # type: ignore[attr-defined]
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class DistilCLIPModule(CLIPModule):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        image_encoder: nn.Module,
        teacher_image_encoder: nn.Module,
        image_proj: nn.Module,
        image_train_part: float,
        text_encoder: nn.Module,
        teacher_text_encoder: nn.Module,
        text_proj: nn.Module,
        text_train_part: float,
        optimizer: Optimizer,
        scheduler: Optional[Scheduler],
        label_smoothing: Optional[float],
        batch_size: int,
    ) -> None:
        super().__init__(
            image_encoder=image_encoder,
            image_train_part=image_train_part,
            text_encoder=text_encoder,
            text_train_part=text_train_part,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smoothing=label_smoothing,
            batch_size=batch_size,
        )

        self.teacher = Clip(image_encoder=teacher_image_encoder, text_encoder=teacher_text_encoder)
        freeze_model(self.teacher, full=True)

        self.image_proj = image_proj
        self.text_proj = text_proj

        self.distil_criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, batch: BatchEncoding) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        images = batch.pop("images")
        teacher_batch = BatchEncoding()
        for key in sorted(batch.keys()):
            if key.startswith("teacher_"):
                teacher_batch[key.replace("teacher_", "")] = batch.pop(key)
        teacher_image_embeddings = self.teacher.normalize(self.teacher.image_encoder(images).pooler_output)
        teacher_text_embeddings = self.teacher.normalize(self.teacher.text_encoder(**teacher_batch).pooler_output)

        image_embeddings = self.clip.normalize(self.clip.image_encoder(images).logits)
        text_embeddings = self.clip.normalize(self.clip.text_encoder(**batch).logits)

        image_logits, text_logits = self.clip(image_embeddings=image_embeddings, text_embeddings=text_embeddings)

        return (
            image_embeddings,
            text_embeddings,
            teacher_image_embeddings,
            teacher_text_embeddings,
            image_logits,
            text_logits,
        )

    def model_step(self, batch: BatchEncoding) -> Tuple[Tensor, Tensor, Tensor]:
        (
            image_embeddings,
            text_embeddings,
            teacher_image_embeddings,
            teacher_text_embeddings,
            image_logits,
            text_logits,
        ) = self.forward(batch=batch)

        labels = arange(image_logits.size(0), device=self.device)
        loss = self.criterion(image_logits, labels) + self.criterion(text_logits, labels)

        image_distrib = F.log_softmax(self.image_proj(image_embeddings), dim=1)
        image_target = F.softmax(teacher_image_embeddings, dim=1)
        text_distrib = F.log_softmax(self.text_proj(text_embeddings), dim=1)
        text_target = F.softmax(teacher_text_embeddings, dim=1)

        loss += 10 * self.distil_criterion(image_distrib, image_target)
        loss += 10 * self.distil_criterion(text_distrib, text_target)

        predict = argmax(image_logits, dim=-1)
        return loss, predict, labels
