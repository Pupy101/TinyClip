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
        train_path: str,
        val_path: str,
        test_path: str,
        tokenizer: str,
        max_length: Optional[int],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()

        df_train = pd.read_csv(train_path, sep="\t")
        self.data_train = CLIPDataset(df_train, transform=create_augmentations("train"))

        df_val = pd.read_csv(val_path, sep="\t")
        self.data_val = CLIPDataset(df_val, transform=create_augmentations("val"))

        df_test = pd.read_csv(test_path, sep="\t")
        self.data_test = CLIPDataset(df_test, transform=create_augmentations("test"))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_len = max_length

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(tokenizer=self.tokenizer, max_length=self.max_len),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(tokenizer=self.tokenizer, max_length=self.max_len),
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(tokenizer=self.tokenizer, max_length=self.max_len),
            drop_last=True,
        )
