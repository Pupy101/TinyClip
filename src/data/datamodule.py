from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from src.data.collate_fn import create_collate_fn, create_collate_with_teacher_fn
from src.data.dataset import CLIPDataset
from src.data.transform import ImageTransform


class CLIPDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,  # pylint: disable=unused-argument
        val_path: str,  # pylint: disable=unused-argument
        test_path: str,  # pylint: disable=unused-argument
        tokenizer: str,
        max_length: Optional[int],  # pylint: disable=unused-argument
        batch_size: int,  # pylint: disable=unused-argument
        num_workers: int,  # pylint: disable=unused-argument
        mean: Sequence[float],
        std: Sequence[float],
        crop: int,
        size: int,
        teacher_tokenizer: Optional[str] = None,
        teacher_max_length: Optional[int] = None,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["tokenizer", "mean", "std", "crop", "size"])

        self.data_train: Optional[CLIPDataset] = None
        self.data_val: Optional[CLIPDataset] = None
        self.data_test: Optional[CLIPDataset] = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer) if teacher_tokenizer else None
        self.augmentation = ImageTransform(mean=mean, std=std, crop=crop, size=size)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.data_train is None:
            df_train = pd.read_csv(self.hparams.train_path, sep="\t", dtype={"ru_text": str, "en_text": str})
            self.data_train = CLIPDataset(df_train, transform=self.augmentation.create("train"))
        if stage in {"fit", "validate"} and self.data_val is None:
            df_val = pd.read_csv(self.hparams.val_path, sep="\t", dtype={"ru_text": str, "en_text": str})
            self.data_val = CLIPDataset(df_val, transform=self.augmentation.create("val"))
        if stage == "test" and self.data_test is None:
            df_test = pd.read_csv(self.hparams.test_path, sep="\t", dtype={"ru_text": str, "en_text": str})
            self.data_test = CLIPDataset(df_test, transform=self.augmentation.create("test"))

    def create_collate(self) -> Callable[[Iterable[Tuple[Optional[Tensor], List[str]]]], BatchEncoding]:
        if self.teacher_tokenizer:
            collate_fn = create_collate_with_teacher_fn(
                tokenizer=self.tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
                max_length=self.hparams.max_length,
                teacher_max_length=self.hparams.teacher_max_length,
            )
        else:
            collate_fn = create_collate_fn(tokenizer=self.tokenizer, max_length=self.hparams.max_length)
        return collate_fn

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.create_collate(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data_val is not None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.create_collate(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data_test is not None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.create_collate(),
        )
