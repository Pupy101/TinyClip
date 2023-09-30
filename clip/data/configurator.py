import abc
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from utilities.data import train_valid_test_split

from clip.data.augmentations import create_image_augs, create_text_augs
from clip.data.collate_fn import create_clip_collate_fn, create_image_collate_fn, create_masked_lm_collate_fn
from clip.data.dataset import CLIPDataset, ImageDataset, TextDataset
from clip.types import DataLoaders, DatasetType, PathLike, Tokenizer


@dataclass
class BaseConfigurator:
    dataframe: PathLike
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int
    train_size: float
    valid_size: float
    test_size: float
    num_workers: int
    tokenizer: str
    max_length: Optional[int]

    def __post_init__(self) -> None:
        assert self.train_size + self.valid_size + self.test_size == 1.0

    @staticmethod
    def read_dataframe(file: PathLike) -> pd.DataFrame:
        return pd.read_table(file)

    @abc.abstractmethod
    def create_loaders(self) -> DataLoaders:
        raise NotImplementedError


@dataclass
class ImageConfigurator(BaseConfigurator):
    image_column: str = "image"
    text_column: str = "text"

    def create_dataset(self, dataframe: pd.DataFrame, dataset_type: str) -> ImageDataset:
        return ImageDataset(
            dataframe,
            image_transform=create_image_augs(dataset_type),
            image_column=self.image_column,
            text_column=self.text_column,
        )

    def create_dataloader(
        self,
        dataset: ImageDataset,
        tokenizer: Tokenizer,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=create_image_collate_fn(tokenizer=tokenizer, max_length=self.max_length),
        )

    def create_loaders(self) -> DataLoaders:
        tokenizer = BertTokenizerFast(self.tokenizer)
        df = self.read_dataframe(self.dataframe)
        train_df, valid_df, test_df = train_valid_test_split(df, valid_size=self.valid_size, test_size=self.test_size)
        train_loader = self.create_dataloader(
            dataset=self.create_dataset(train_df, dataset_type=DatasetType.TRAIN.value),
            tokenizer=tokenizer,
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        valid_loader = self.create_dataloader(
            dataset=self.create_dataset(valid_df, dataset_type=DatasetType.VALID.value),
            tokenizer=tokenizer,
            batch_size=self.valid_batch_size,
            shuffle=False,
        )
        test_loader = self.create_dataloader(
            dataset=self.create_dataset(test_df, dataset_type=DatasetType.TEST.value),
            tokenizer=tokenizer,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader)


@dataclass
class TextConfigurator(BaseConfigurator):
    text_column: str = "text"

    def create_dataset(self, dataframe: pd.DataFrame, dataset_type: str) -> TextDataset:
        return TextDataset(
            dataframe,
            text_transform=create_text_augs(dataset_type),
            text_column=self.text_column,
        )

    def create_dataloader(
        self,
        dataset: TextDataset,
        batch_size: int,
        tokenizer: Tokenizer,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=create_masked_lm_collate_fn(tokenizer=tokenizer, max_length=self.max_length),
        )

    def create_loaders(self) -> DataLoaders:
        tokenizer = BertTokenizerFast(self.tokenizer)
        df = self.read_dataframe(self.dataframe)
        train_df, valid_df, test_df = train_valid_test_split(df, valid_size=self.valid_size, test_size=self.test_size)
        train_loader = self.create_dataloader(
            dataset=self.create_dataset(train_df, dataset_type=DatasetType.TRAIN.value),
            batch_size=self.train_batch_size,
            tokenizer=tokenizer,
            shuffle=True,
        )
        valid_loader = self.create_dataloader(
            dataset=self.create_dataset(valid_df, dataset_type=DatasetType.VALID.value),
            batch_size=self.valid_batch_size,
            tokenizer=tokenizer,
            shuffle=False,
        )
        test_loader = self.create_dataloader(
            dataset=self.create_dataset(test_df, dataset_type=DatasetType.TEST.value),
            batch_size=self.test_batch_size,
            tokenizer=tokenizer,
            shuffle=False,
        )
        return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader)


@dataclass
class CLIPConfigurator(BaseConfigurator):
    image_column: str = "image"
    text_column: str = "text"

    def create_dataset(self, dataframe: pd.DataFrame, dataset_type: str) -> CLIPDataset:
        return CLIPDataset(
            dataframe,
            image_transform=create_image_augs(dataset_type),
            text_transform=create_text_augs(dataset_type),
            image_column=self.image_column,
            text_column=self.text_column,
        )

    def create_dataloader(
        self,
        dataset: CLIPDataset,
        batch_size: int,
        tokenizer: Tokenizer,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=create_clip_collate_fn(tokenizer=tokenizer, max_length=self.max_length),
        )

    def create_loaders(self) -> DataLoaders:
        tokenizer = BertTokenizerFast(self.tokenizer)
        df = self.read_dataframe(self.dataframe)
        train_df, valid_df, test_df = train_valid_test_split(df, valid_size=self.valid_size, test_size=self.test_size)
        train_loader = self.create_dataloader(
            dataset=self.create_dataset(train_df, dataset_type=DatasetType.TRAIN.value),
            batch_size=self.train_batch_size,
            tokenizer=tokenizer,
            shuffle=True,
        )
        valid_loader = self.create_dataloader(
            dataset=self.create_dataset(valid_df, dataset_type=DatasetType.VALID.value),
            batch_size=self.valid_batch_size,
            tokenizer=tokenizer,
            shuffle=False,
        )
        test_loader = self.create_dataloader(
            dataset=self.create_dataset(test_df, dataset_type=DatasetType.TEST.value),
            batch_size=self.test_batch_size,
            tokenizer=tokenizer,
            shuffle=False,
        )
        return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader)
