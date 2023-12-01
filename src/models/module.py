from typing import Dict, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import BatchEncoding, CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput

from src.models.clip import Clip
from src.types import Optimizer, Scheduler
from src.utils import freeze_model


class CLIPModule(LightningModule):
    def __init__(
        self,
        img: nn.Module,
        img_part: float,
        txt: nn.Module,
        txt_part: float,
        optimizer: Optimizer,  # pylint: disable=unused-argument
        scheduler: Optional[Scheduler],  # pylint: disable=unused-argument
        threshold: Optional[float],  # pylint: disable=unused-argument
        bce_coeff: float,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()

        self.save_hyperparameters("optimizer", "scheduler", "threshold", "bce_coeff", logger=False)

        freeze_model(model=img, train_part=img_part)
        freeze_model(model=txt, train_part=txt_part)

        self.clip = Clip(img=img, txt=txt)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.acc_train = BinaryAccuracy()
        self.acc_val = BinaryAccuracy()
        self.acc_test = BinaryAccuracy()

        self.f1_train = BinaryF1Score()
        self.f1_val = BinaryF1Score()
        self.f1_test = BinaryF1Score()

    def forward(self, batch: BatchEncoding) -> Dict[str, Tensor]:
        tensors: Dict[str, Tensor] = {}
        tensors["img_input"] = batch.pop("pixel_values")
        tensors["txt_input"] = batch

        img_emb = self.clip.normalize(self.clip.img(tensors["img_input"]).logits)
        txt_emb = self.clip.normalize(self.clip.txt(**tensors["txt_input"]).logits)  # type: ignore
        tensors.update({"img_emb": img_emb, "txt_emb": txt_emb})

        img_logit, txt_logit = self.clip(img=img_emb, txt=txt_emb)
        tensors.update({"img_logit": img_logit, "txt_logit": txt_logit})

        return tensors

    def model_step(self, batch: BatchEncoding) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = self.forward(batch=batch)

        if "label_bce" not in tensors:
            label_bce = torch.diag(torch.ones(tensors["img_logit"].size(0), device=self.device))
            if self.hparams.threshold is not None:
                txt_sim = tensors["txt_emb"] @ tensors["txt_emb"].T
                label_bce[txt_sim > self.hparams.threshold] = 1.0
            tensors.update({"label_bce": label_bce})
        else:
            label_bce = tensors["label_bce"]

        predict = torch.sigmoid(tensors["img_logit"])
        tensors.update({"predict": predict})

        img_bce_loss = self.hparams.bce_coeff * self.bce_loss(tensors["img_logit"], label_bce)
        txt_bce_loss = self.hparams.bce_coeff * self.bce_loss(tensors["txt_logit"], label_bce)
        losses = {"img_bce_loss": img_bce_loss, "txt_bce_loss": txt_bce_loss, "loss": img_bce_loss + txt_bce_loss}

        label_ce = torch.arange(tensors["img_logit"].size(0), device=self.device, dtype=torch.long)
        img_ce_loss = self.ce_loss(tensors["img_logit"], label_ce)
        txt_ce_loss = self.ce_loss(tensors["txt_logit"], label_ce)
        losses["loss"] += img_ce_loss + txt_ce_loss
        tensors.update({"label_ce": label_ce})
        losses.update({"img_ce_loss": img_ce_loss, "txt_ce_loss": txt_ce_loss})

        return tensors, losses

    def on_train_start(self) -> None:
        self.acc_val.reset()
        self.f1_val.reset()

    def compute_metric(
        self, prefix: str, tensors: Dict[str, Tensor], losses: Dict[str, Tensor], acc_metric: Metric, f1_metric: Metric
    ) -> None:
        acc = acc_metric(tensors["predict"].flatten(), tensors["label_bce"].flatten())
        f1 = f1_metric(tensors["predict"].flatten(), tensors["label_bce"].flatten())

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
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "interval": "step", "frequency": 1},
            }
        return {"optimizer": optimizer}


class DistilCLIPModule(CLIPModule):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        img: nn.Module,
        img_proj: nn.Module,
        img_part: float,
        txt: nn.Module,
        txt_proj: nn.Module,
        txt_part: float,
        teacher: CLIPModel,  # pylint: disable=unused-argument
        optimizer: Optimizer,
        scheduler: Optional[Scheduler],
        threshold: float,
        bce_coeff: float,
        l1_coeff: float,  # pylint: disable=unused-argument
        cos_coeff: float,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(
            img=img,
            img_part=img_part,
            txt=txt,
            txt_part=txt_part,
            optimizer=optimizer,
            scheduler=scheduler,
            threshold=threshold,
            bce_coeff=bce_coeff,
        )
        self.save_hyperparameters("l1_coeff", "cos_coeff", logger=False)

        self.teacher = teacher
        self.teacher.eval()
        freeze_model(self.teacher, full=True)

        self.img_proj = img_proj
        self.txt_proj = txt_proj

        self.l1_loss = nn.L1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.1)

    def forward(self, batch: BatchEncoding) -> Dict[str, Tensor]:  # type: ignore[override]
        inputs = BatchEncoding()
        for key in sorted(batch.keys()):
            if key.startswith("tchr_"):
                inputs[key.replace("tchr_", "")] = batch.pop(key)

        tensors = super().forward(batch=batch)
        out: CLIPOutput = self.teacher(**inputs)

        label_bce = torch.diag(torch.ones(out.text_embeds.size(0), device=self.device))
        if self.hparams.threshold is not None:
            txt_sim = out.text_embeds @ out.text_embeds.T
            label_bce[txt_sim > self.hparams.threshold] = 1.0
        batch.update({"label_bce": label_bce})

        tensors.update({"tchr_txt_input": inputs, "tchr_img_emb": out.image_embeds, "tchr_txt_emb": out.text_embeds})

        return tensors

    def model_step(self, batch: BatchEncoding) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tensors, losses = super().model_step(batch=batch)

        img_proj_emb = self.clip.normalize(self.img_proj(tensors["img_emb"]))
        txt_proj_emb = self.clip.normalize(self.txt_proj(tensors["txt_emb"]))
        tensors.update({"img_proj_emb": img_proj_emb, "txt_proj_emb": txt_proj_emb})

        img_l1_loss = self.hparams.l1_coeff * self.l1_loss(img_proj_emb, tensors["tchr_img_emb"])
        txt_l1_loss = self.hparams.l1_coeff * self.l1_loss(txt_proj_emb, tensors["tchr_txt_emb"])
        losses["loss"] += img_l1_loss + txt_l1_loss
        losses.update({"img_l1_loss": img_l1_loss, "txt_l1_loss": txt_l1_loss})

        img_cos_loss = self.hparams.cos_coeff * self.cos_loss(
            img_proj_emb, tensors["tchr_img_emb"], tensors["label_ce"]
        )
        txt_cos_loss = self.hparams.cos_coeff * self.cos_loss(
            txt_proj_emb, tensors["tchr_txt_emb"], tensors["label_ce"]
        )
        losses["loss"] += img_cos_loss + txt_cos_loss
        losses.update({"img_cos_loss": img_cos_loss, "txt_cos_loss": txt_cos_loss})

        return tensors, losses
