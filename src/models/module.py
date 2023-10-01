from typing import Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn, ones, sigmoid, zeros
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score
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
        pos_weight: float,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["image_encoder", "text_encoder"])

        self.clip = Clip(image_encoder=image_encoder, text_encoder=text_encoder)
        self.pos_weight = pos_weight

        self.loss_train = MeanMetric()
        self.loss_val = MeanMetric()
        self.loss_test = MeanMetric()

        self.f1_train = BinaryF1Score()
        self.f1_val = BinaryF1Score()
        self.f1_test = BinaryF1Score()

    def forward(self, batch: BatchEncoding) -> Tuple[Tensor, Tensor]:
        images = batch.pop("images")
        image_embeddings = self.clip.normalize(self.clip.image_encoder(images).logits)
        text_embeddings = self.clip.normalize(self.clip.text_encoder(**batch).logits)
        image_logits, text_logits = self.clip(image_embeddings=image_embeddings, text_embeddings=text_embeddings)
        return image_logits, text_logits

    def model_step(self, batch: BatchEncoding) -> Tuple[Tensor, Tensor, Tensor]:
        image_indexes, text_indexes = batch.pop("image_indexes"), batch.pop("text_indexes")
        image_logits, text_logits = self.forward(batch=batch)
        labels = zeros(image_logits.size(0), text_logits.size(0), device=self.device)
        labels[image_indexes, text_indexes] = 1
        pos_weight = self.pos_weight * ones(labels.size(1), device=self.device)
        image_loss = F.binary_cross_entropy_with_logits(image_logits, labels, pos_weight=pos_weight)
        pos_weight = self.pos_weight * ones(labels.size(0), device=self.device)
        text_loss = F.binary_cross_entropy_with_logits(text_logits, labels.t(), pos_weight=pos_weight)
        probas = sigmoid(image_logits.reshape(-1))
        return image_loss + text_loss, probas, labels.reshape(-1)

    def on_train_start(self) -> None:
        self.loss_val.reset()
        self.f1_val.reset()

    def training_step(self, batch: BatchEncoding, _: int) -> Tensor:
        loss, probas, labels = self.model_step(batch)

        self.loss_train(loss)
        f1 = self.f1_train(probas, labels)

        self.log("train/loss", self.loss_train, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: BatchEncoding, _: int) -> None:
        loss, probas, labels = self.model_step(batch)

        self.loss_val(loss)
        f1 = self.f1_val(probas, labels)

        self.log("val/loss", self.loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: BatchEncoding, _: int) -> None:
        loss, probas, labels = self.model_step(batch)

        self.loss_test(loss)
        f1 = self.f1_test(probas, labels)

        self.log("test/loss", self.loss_test, on_step=True, on_epoch=True, prog_bar=True)
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
