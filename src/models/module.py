from typing import Dict, Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, arange, argmax, long, nn
from torchmetrics import Metric
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
        num_classes: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters("optimizer", "scheduler", logger=False)

        freeze_model(model=image_encoder, train_part=image_train_part)
        freeze_model(model=text_encoder, train_part=text_train_part)

        self.clip = Clip(image_encoder=image_encoder, text_encoder=text_encoder)

        self.ce_loss = nn.CrossEntropyLoss()

        self.acc_train = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.acc_val = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.acc_test = MulticlassAccuracy(num_classes=num_classes, average="weighted")

        self.f1_train = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.f1_val = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.f1_test = MulticlassF1Score(num_classes=num_classes, average="weighted")

    def forward(self, batch: BatchEncoding) -> Dict[str, Tensor]:
        tensors: Dict[str, Tensor] = {}
        tensors["img_input"] = batch.pop("image")
        tensors["txt_input"] = batch
        img_emb = self.clip.normalize(self.clip.image_encoder(tensors["img_input"]).logits)
        txt_emb = self.clip.normalize(self.clip.text_encoder(**tensors["txt_input"]).logits)
        tensors.update({"img_emb": img_emb, "txt_emb": txt_emb})
        img_logit, txt_logit = self.clip(image_embeddings=img_emb, text_embeddings=txt_emb)
        tensors.update({"img_logit": img_logit, "txt_logit": txt_logit})
        return tensors

    def model_step(self, batch: BatchEncoding) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = self.forward(batch=batch)
        if "label" in batch:
            label = batch.pop("label")
        else:
            label = arange(tensors["img_logit"].size(0), dtype=long, device=self.device)
        predict = argmax(tensors["img_logit"], dim=-1)
        tensors.update({"label": label, "predict": predict})
        img_loss = self.ce_loss(tensors["img_logit"], label)
        txt_loss = self.ce_loss(tensors["txt_logit"], label)
        losses = {"img_ce_loss": img_loss, "txt_ce_loss": txt_loss, "loss": img_loss + txt_loss}
        return tensors, losses

    def on_train_start(self) -> None:
        self.acc_val.reset()
        self.f1_val.reset()

    def compute_metric(
        self,
        prefix: str,
        tensors: Dict[str, Tensor],
        losses: Dict[str, Tensor],
        acc_metric: Metric,
        f1_metric: Metric,
    ) -> None:
        acc = acc_metric(tensors["predict"], tensors["label"])
        f1 = f1_metric(tensors["predict"], tensors["label"])

        for key, value in losses.items():
            if key == "loss":
                self.log(prefix + f"/{key}", value.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                self.log(prefix + f"/{key}", value.item(), on_epoch=True, sync_dist=True)
        self.log(prefix + "/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(prefix + "/f1", f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch: BatchEncoding, _: int) -> Tensor:
        tensors, losses = self.model_step(batch)
        self.compute_metric(
            prefix="train", tensors=tensors, losses=losses, acc_metric=self.acc_train, f1_metric=self.f1_train
        )
        return losses["loss"]

    def validation_step(self, batch: BatchEncoding, _: int) -> Tensor:
        tensors, losses = self.model_step(batch)
        self.compute_metric(
            prefix="val", tensors=tensors, losses=losses, acc_metric=self.acc_val, f1_metric=self.f1_val
        )
        return losses["loss"]

    def test_step(self, batch: BatchEncoding, _: int) -> Tensor:
        tensors, losses = self.model_step(batch)
        self.compute_metric(
            prefix="test", tensors=tensors, losses=losses, acc_metric=self.acc_test, f1_metric=self.f1_test
        )
        return losses["loss"]

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
        num_classes: int,
    ) -> None:
        super().__init__(
            image_encoder=image_encoder,
            image_train_part=image_train_part,
            text_encoder=text_encoder,
            text_train_part=text_train_part,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=num_classes,
        )

        self.teacher = Clip(image_encoder=teacher_image_encoder, text_encoder=teacher_text_encoder)
        freeze_model(self.teacher, full=True)

        self.image_proj = image_proj
        self.text_proj = text_proj

        self.l1_loss = nn.L1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.25)

    def forward(self, batch: BatchEncoding) -> Dict[str, Tensor]:  # type: ignore[override]
        txt_input = BatchEncoding()
        for key in sorted(batch.keys()):
            if key.startswith("teacher_"):
                txt_input[key.replace("teacher_", "")] = batch.pop(key)

        tensors = super().forward(batch=batch)

        img_emb = self.teacher.normalize(self.teacher.image_encoder(tensors["img_input"]).pooler_output)
        txt_emb = self.teacher.normalize(self.teacher.text_encoder(**txt_input).pooler_output)
        tensors.update({"teacher_txt_input": txt_input, "teacher_img_emb": img_emb, "teacher_txt_emb": txt_emb})

        return tensors

    def model_step(self, batch: BatchEncoding) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tensors, losses = super().model_step(batch=batch)

        img_proj_emb = self.clip.normalize(self.image_proj(tensors["img_emb"]))
        txt_proj_emb = self.clip.normalize(self.text_proj(tensors["txt_emb"]))
        tensors.update({"img_proj_emb": img_proj_emb, "txt_proj_emb": txt_proj_emb})

        img_l1_loss = 5 * self.l1_loss(img_proj_emb, tensors["teacher_img_emb"])
        txt_l1_loss = 5 * self.l1_loss(txt_proj_emb, tensors["teacher_txt_emb"])
        losses["loss"] += img_l1_loss + txt_l1_loss
        losses.update({"img_l1_loss": img_l1_loss, "txt_l1_loss": txt_l1_loss})

        img_cos_loss = 5 * self.cos_loss(img_proj_emb, tensors["teacher_img_emb"], tensors["label"])
        txt_cos_loss = 5 * self.cos_loss(txt_proj_emb, tensors["teacher_txt_emb"], tensors["label"])
        losses["loss"] += img_cos_loss + txt_cos_loss
        losses.update({"img_cos_loss": img_cos_loss, "txt_cos_loss": txt_cos_loss})

        return tensors, losses
