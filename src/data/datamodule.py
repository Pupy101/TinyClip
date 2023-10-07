from typing import Optional

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.augmentations import create_augmentations
from src.data.collate_fn import create_collate_fn
from src.data.dataset import CLIPDataset


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
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore="tokenizer")

        self.data_train: Optional[CLIPDataset] = None
        self.data_val: Optional[CLIPDataset] = None
        self.data_test: Optional[CLIPDataset] = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.data_train is None:
            df_train = pd.read_csv(self.hparams.train_path, sep="\t", dtype={"ru_text": str, "en_text": str})
            self.data_train = CLIPDataset(df_train, transform=create_augmentations("train"))
        if stage in {"fit", "validate"} and self.data_val is None:
            df_val = pd.read_csv(self.hparams.val_path, sep="\t", dtype={"ru_text": str, "en_text": str})
            self.data_val = CLIPDataset(df_val, transform=create_augmentations("val"))
        if stage == "test" and self.data_test is None:
            df_test = pd.read_csv(self.hparams.test_path, sep="\t", dtype={"ru_text": str, "en_text": str})
            self.data_test = CLIPDataset(df_test, transform=create_augmentations("test"))

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=create_collate_fn(self.tokenizer, max_length=self.hparams.max_length),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data_val is not None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=create_collate_fn(self.tokenizer, max_length=self.hparams.max_length),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data_test is not None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=create_collate_fn(self.tokenizer, max_length=self.hparams.max_length),
        )
