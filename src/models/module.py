from typing import Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, arange, argmax, nn, sigmoid
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import BatchEncoding

from src.models.clip import Clip
from src.types import Optimizer, Scheduler


class CLIPModule(LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        optimizer: Optimizer,  # pylint: disable=unused-argument
        scheduler: Optional[Scheduler],  # pylint: disable=unused-argument
        label_smoothing: Optional[float],
        batch_size: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["image_encoder", "text_encoder", "label_smoothing", "batch_size"],
        )

        self.clip = Clip(image_encoder=image_encoder, text_encoder=text_encoder)

        label_smoothing = label_smoothing or 0.0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.batch_size = batch_size

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
        probas = sigmoid(image_logits) if image_logits.size(1) == self.batch_size else argmax(image_logits, dim=-1)
        return loss, probas, labels

    def on_train_start(self) -> None:
        self.loss_val.reset()
        self.acc_val.reset()
        self.f1_val.reset()

    def training_step(self, batch: BatchEncoding, _: int) -> Tensor:
        loss, probas, labels = self.model_step(batch)

        self.loss_train(loss)
        acc = self.acc_train(probas, labels)
        f1 = self.f1_train(probas, labels)

        self.log("train/loss", self.loss_train, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: BatchEncoding, _: int) -> None:
        loss, probas, labels = self.model_step(batch)

        self.loss_val(loss)
        acc = self.acc_val(probas, labels)
        f1 = self.f1_val(probas, labels)

        self.log("val/loss", self.loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: BatchEncoding, _: int) -> None:
        loss, probas, labels = self.model_step(batch)

        self.loss_test(loss)
        acc = self.acc_test(probas, labels)
        f1 = self.f1_test(probas, labels)

        self.log("test/loss", self.loss_test, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

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
