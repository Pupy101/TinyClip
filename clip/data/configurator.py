from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from clip.types import BatchSizes, DataLoaders, DatasetType, SplitSizes, Tokenizer
from clip.utils.functions import split_train_val_test, split_train_val_test_stratify

from .augmentations import create_image_augs, create_text_augs
from .collate_fn import create_clip_collate_fn, create_masked_lm_collate_fn
from .dataset import ClassificationDataset, CLIPDataset, MaskedLMDataset


class Configurator:
    @staticmethod
    def create_clip_dataset(
        dataframe: pd.DataFrame,
        dataset_type: str,
        image_column: str = "image",
        text_column: str = "text",
    ) -> CLIPDataset:
        return CLIPDataset(
            dataframe=dataframe,
            image_transform=create_image_augs(dataset_type),
            text_transform=create_text_augs(dataset_type),
            image_column=image_column,
            text_column=text_column,
        )

    @staticmethod
    def create_clip_dataloader(  # pylint: disable=too-many-arguments
        dataset: CLIPDataset,
        batch_size: int,
        num_workers: int,
        tokenizer: Tokenizer,
        shuffle: bool,
        max_length: Optional[int] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=create_clip_collate_fn(tokenizer=tokenizer, max_length=max_length),
        )

    @staticmethod
    def create_image_dataset(
        dataframe: pd.DataFrame,
        dataset_type: str,
        image_column: str = "image",
        label_column: str = "label",
    ) -> ClassificationDataset:
        return ClassificationDataset(
            dataframe=dataframe,
            image_transform=create_image_augs(dataset_type),
            image_column=image_column,
            label_column=label_column,
        )

    @staticmethod
    def create_image_dataloader(  # pylint: disable=too-many-arguments
        dataset: ClassificationDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    @staticmethod
    def create_text_dataset(
        dataframe: pd.DataFrame,
        dataset_type: str,
        text_column: str = "text",
    ) -> MaskedLMDataset:
        return MaskedLMDataset(
            dataframe=dataframe,
            text_transform=create_text_augs(dataset_type),
            text_column=text_column,
        )

    @staticmethod
    def create_text_dataloader(  # pylint: disable=too-many-arguments
        dataset: MaskedLMDataset,
        batch_size: int,
        num_workers: int,
        tokenizer: Tokenizer,
        shuffle: bool,
        max_length: Optional[int] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=create_masked_lm_collate_fn(tokenizer=tokenizer, max_length=max_length),
        )

    @staticmethod
    def create_clip_dataloaders(  # pylint: disable=too-many-arguments
        dataframe: pd.DataFrame,
        batch_sizes: BatchSizes,
        split_sizes: SplitSizes,
        num_workers: int,
        tokenizer: Tokenizer,
        image_column: str = "image",
        text_column: str = "text",
        max_length: Optional[int] = None,
    ) -> DataLoaders:
        train_df, valid_df, test_df = split_train_val_test(
            dataframe,
            train_size=split_sizes.train,
            valid_size=split_sizes.test,
        )
        train_loader = Configurator.create_clip_dataloader(
            dataset=Configurator.create_clip_dataset(
                train_df,
                dataset_type=DatasetType.TRAIN.value,
                image_column=image_column,
                text_column=text_column,
            ),
            batch_size=batch_sizes.train,
            num_workers=num_workers,
            tokenizer=tokenizer,
            shuffle=True,
            max_length=max_length,
        )
        valid_loader = Configurator.create_clip_dataloader(
            dataset=Configurator.create_clip_dataset(
                valid_df,
                dataset_type=DatasetType.VALID.value,
                image_column=image_column,
                text_column=text_column,
            ),
            batch_size=batch_sizes.valid,
            num_workers=num_workers,
            tokenizer=tokenizer,
            shuffle=False,
            max_length=max_length,
        )
        test_loader = Configurator.create_clip_dataloader(
            dataset=Configurator.create_clip_dataset(
                test_df,
                dataset_type=DatasetType.TEST.value,
                image_column=image_column,
                text_column=text_column,
            ),
            batch_size=batch_sizes.test,
            num_workers=num_workers,
            tokenizer=tokenizer,
            shuffle=False,
            max_length=max_length,
        )
        return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader)

    @staticmethod
    def create_image_dataloaders(  # pylint: disable=too-many-arguments
        dataframe: pd.DataFrame,
        batch_sizes: BatchSizes,
        split_sizes: SplitSizes,
        num_workers: int,
        image_column: str = "image",
        label_column: str = "label",
    ) -> DataLoaders:
        train_df, valid_df, test_df = split_train_val_test_stratify(
            dataframe,
            train_size=split_sizes.train,
            valid_size=split_sizes.test,
            stratify_column=label_column,
        )
        train_loader = Configurator.create_image_dataloader(
            dataset=Configurator.create_image_dataset(
                train_df,
                dataset_type=DatasetType.TRAIN.value,
                image_column=image_column,
                label_column=label_column,
            ),
            batch_size=batch_sizes.train,
            num_workers=num_workers,
            shuffle=True,
        )
        valid_loader = Configurator.create_image_dataloader(
            dataset=Configurator.create_image_dataset(
                valid_df,
                dataset_type=DatasetType.VALID.value,
                image_column=image_column,
                label_column=label_column,
            ),
            batch_size=batch_sizes.valid,
            num_workers=num_workers,
            shuffle=False,
        )
        test_loader = Configurator.create_image_dataloader(
            dataset=Configurator.create_image_dataset(
                test_df,
                dataset_type=DatasetType.TEST.value,
                image_column=image_column,
                label_column=label_column,
            ),
            batch_size=batch_sizes.test,
            num_workers=num_workers,
            shuffle=False,
        )
        return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader)

    @staticmethod
    def create_text_dataloaders(  # pylint: disable=too-many-arguments
        dataframe: pd.DataFrame,
        batch_sizes: BatchSizes,
        split_sizes: SplitSizes,
        num_workers: int,
        tokenizer: Tokenizer,
        text_column: str = "text",
        max_length: Optional[int] = None,
    ) -> DataLoaders:
        train_df, valid_df, test_df = split_train_val_test(
            dataframe,
            train_size=split_sizes.train,
            valid_size=split_sizes.test,
        )
        train_loader = Configurator.create_text_dataloader(
            dataset=Configurator.create_text_dataset(
                train_df,
                dataset_type=DatasetType.TRAIN.value,
                text_column=text_column,
            ),
            batch_size=batch_sizes.train,
            num_workers=num_workers,
            tokenizer=tokenizer,
            shuffle=True,
            max_length=max_length,
        )
        valid_loader = Configurator.create_text_dataloader(
            dataset=Configurator.create_text_dataset(
                valid_df,
                dataset_type=DatasetType.VALID.value,
                text_column=text_column,
            ),
            batch_size=batch_sizes.valid,
            num_workers=num_workers,
            tokenizer=tokenizer,
            shuffle=False,
            max_length=max_length,
        )
        test_loader = Configurator.create_text_dataloader(
            dataset=Configurator.create_text_dataset(
                test_df,
                dataset_type=DatasetType.TEST.value,
                text_column=text_column,
            ),
            batch_size=batch_sizes.test,
            num_workers=num_workers,
            tokenizer=tokenizer,
            shuffle=False,
            max_length=max_length,
        )
        return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader)
