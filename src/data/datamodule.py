from typing import Callable, Iterable, Optional, Tuple

import pandas as pd
from lightning import LightningDataModule
from PIL.Image import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer, BatchEncoding

from src.data.collate_fn import create_collate_fn, create_distil_collate_fn, create_transform
from src.data.dataset import CLIPDataset


class CLIPDataModule(LightningDataModule):
    def __init__(
        self,
        path_train: str,  # pylint: disable=unused-argument
        path_val: str,  # pylint: disable=unused-argument
        path_test: str,  # pylint: disable=unused-argument
        processor: str,
        tokenizer: str,
        max_length: Optional[int],  # pylint: disable=unused-argument
        batch_size: int,  # pylint: disable=unused-argument
        num_workers: int,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["processor", "tokenizer"], logger=False)

        self.data_train: Optional[CLIPDataset] = None
        self.data_val: Optional[CLIPDataset] = None
        self.data_test: Optional[CLIPDataset] = None

        self.processor = AutoImageProcessor.from_pretrained(processor)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.data_train is None:
            df_train = pd.read_csv(self.hparams.path_train, sep="\t")
            self.data_train = CLIPDataset(df_train)
        if stage in {"fit", "validate"} and self.data_val is None:
            df_val = pd.read_csv(self.hparams.path_val, sep="\t")
            self.data_val = CLIPDataset(df_val)
        if stage == "test" and self.data_test is None:
            df_test = pd.read_csv(self.hparams.path_test, sep="\t")
            self.data_test = CLIPDataset(df_test)

    def create_collate(self, is_train: bool = False) -> Callable[[Iterable[Tuple[Image, str]]], BatchEncoding]:
        transform = create_transform() if is_train else None
        return create_collate_fn(
            processor=self.processor, tokenizer=self.tokenizer, max_length=self.hparams.max_length, transform=transform
        )

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.create_collate(is_train=True),
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


class DistilCLIPDataModule(CLIPDataModule):
    def __init__(
        self,
        path_train: str,
        path_val: str,
        path_test: str,
        processor: str,
        tokenizer: str,
        max_length: Optional[int],
        teacher_processor: str,
        teacher_tokenizer: str,
        teacher_max_length: Optional[int],  # pylint: disable=unused-argument
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__(
            path_train=path_train,
            path_val=path_val,
            path_test=path_test,
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.save_hyperparameters("teacher_max_length", logger=False)

        self.teacher_processor = AutoImageProcessor.from_pretrained(teacher_processor)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer)

    def create_collate(self, is_train: bool = False) -> Callable[[Iterable[Tuple[Image, str]]], BatchEncoding]:
        transform = create_transform() if is_train else None
        return create_distil_collate_fn(
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length,
            teacher_processor=self.teacher_processor,
            teacher_tokenizer=self.teacher_tokenizer,
            teacher_max_length=self.hparams.teacher_max_length,
            transform=transform,
        )
