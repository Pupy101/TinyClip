from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader
from youtokentome import BPE

from src.data import CLIPDataset, ImageDataset, MaskedLMDataset
from src.models import CLIP, TextPartCLIP, VisionPartCLIP
from src.types import (
    DataConfig,
    DataLoaders,
    MultiTaskDataLoaders,
    TrainConfig,
    TrainingParameters,
)

from .engine import Engine


class Configurator:
    def __init__(
        self,
        vision_part: VisionPartCLIP,
        text_part: TextPartCLIP,
        data_config: DataConfig,
        train_config: TrainConfig,
    ) -> None:
        self.data_config = data_config
        self.train_config = train_config
        self.vision_part = vision_part
        self.text_part = text_part

    @staticmethod
    def open_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        assert csv_path.is_file(), f"File {str(csv_path)} doesn't exist"
        return pd.read_csv(csv_path)

    def configurate_tokenizer(self) -> BPE:
        checkpoint = Path(self.data_config.text_tokenizer_checkpoint)
        assert checkpoint.exists()
        tokenizer = BPE(model=str(checkpoint))
        return tokenizer

    def configurate_clip_datasets(
        self, tokenizer: BPE
    ) -> Tuple[CLIPDataset, CLIPDataset]:
        train_clip_df = self.open_csv(self.data_config.train_clip_csv)
        valid_clip_df = self.open_csv(self.data_config.valid_clip_csv)
        train_clip_dataset = CLIPDataset(
            dataframe=train_clip_df,
            image_transform=self.data_config.image_augmentations.train,
            tokenizer=tokenizer,
            name_image_column=self.data_config.clip_name_image_column,
            name_text_column=self.data_config.clip_name_text_column,
            cls_token_ids=self.data_config.cls_token_ids,
            pad_token_ids=self.data_config.pad_token_ids,
            tokens_max_len=self.data_config.tokens_max_len,
        )
        valid_clip_dataset = CLIPDataset(
            dataframe=valid_clip_df,
            image_transform=self.data_config.image_augmentations.validation,
            tokenizer=tokenizer,
            name_image_column=self.data_config.clip_name_image_column,
            name_text_column=self.data_config.clip_name_text_column,
            cls_token_ids=self.data_config.cls_token_ids,
            pad_token_ids=self.data_config.pad_token_ids,
            tokens_max_len=self.data_config.tokens_max_len,
        )
        return train_clip_dataset, valid_clip_dataset

    def configurate_image_datasets(self) -> Tuple[ImageDataset, ImageDataset]:
        train_image_df = self.open_csv(self.data_config.train_image_classification_csv)
        valid_image_df = self.open_csv(self.data_config.valid_image_classification_csv)
        train_image_dataset = ImageDataset(
            dataframe=train_image_df,
            transform=self.data_config.image_augmentations.train,
            name_image_column=self.data_config.image_classification_name_image_column,
            name_label_column=self.data_config.image_classification_name_label_column,
        )
        valid_image_dataset = ImageDataset(
            dataframe=valid_image_df,
            transform=self.data_config.image_augmentations.validation,
            name_image_column=self.data_config.image_classification_name_image_column,
            name_label_column=self.data_config.image_classification_name_label_column,
        )
        return train_image_dataset, valid_image_dataset

    def configurate_text_datasets(
        self, tokenizer: BPE
    ) -> Tuple[MaskedLMDataset, MaskedLMDataset]:
        train_text_df = self.open_csv(self.data_config.train_masked_lm_csv)
        valid_text_df = self.open_csv(self.data_config.valid_masked_lm_csv)
        train_text_dataset = MaskedLMDataset(
            dataframe=train_text_df,
            tokenizer=tokenizer,
            transform=self.data_config.mask_text_transform,
            name_text_column=self.data_config.masked_lm_name_text_column,
            mask_token_idx=self.data_config.mask_token_idx,
            pad_token_ids=self.data_config.pad_token_ids,
            tokens_max_len=self.data_config.tokens_max_len,
            masked_portion=self.data_config.masked_portion,
        )
        valid_text_dataset = MaskedLMDataset(
            dataframe=valid_text_df,
            tokenizer=tokenizer,
            transform=self.data_config.mask_text_transform,
            name_text_column=self.data_config.masked_lm_name_text_column,
            mask_token_idx=self.data_config.mask_token_idx,
            pad_token_ids=self.data_config.pad_token_ids,
            tokens_max_len=self.data_config.tokens_max_len,
            masked_portion=self.data_config.masked_portion,
        )
        return train_text_dataset, valid_text_dataset

    def configurate_dataloaders(self) -> MultiTaskDataLoaders:
        tokenizer = self.configurate_tokenizer()
        train_clip_dataset, valid_clip_dataset = self.configurate_clip_datasets(
            tokenizer=tokenizer
        )
        train_image_dataset, valid_image_dataset = self.configurate_image_datasets()
        train_text_dataset, valid_text_dataset = self.configurate_text_datasets(
            tokenizer=tokenizer
        )
        train_clip_loader = DataLoader(
            train_clip_dataset,
            batch_size=self.data_config.clip_batch_size_train,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )
        valid_clip_loader = DataLoader(
            valid_clip_dataset,
            batch_size=self.data_config.clip_batch_size_valid,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        )
        train_image_loader = DataLoader(
            train_image_dataset,
            batch_size=self.data_config.image_classification_batch_size_train,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )
        valid_image_loader = DataLoader(
            valid_image_dataset,
            batch_size=self.data_config.image_classification_batch_size_valid,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        )
        train_text_loader = DataLoader(
            train_text_dataset,
            batch_size=self.data_config.masked_lm_batch_size_train,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )
        valid_text_loader = DataLoader(
            valid_text_dataset,
            batch_size=self.data_config.masked_lm_batch_size_valid,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        )
        clip_dataloaders = DataLoaders(
            train=train_clip_loader, validation=valid_clip_loader
        )
        image_dataloaders = DataLoaders(
            train=train_image_loader, validation=valid_image_loader
        )
        text_dataloaders = DataLoaders(
            train=train_text_loader, validation=valid_text_loader
        )
        return MultiTaskDataLoaders(
            clip=clip_dataloaders, image=image_dataloaders, text=text_dataloaders
        )

    def configurate_engine(self) -> Engine:
        clip = CLIP(
            vision_part=self.vision_part,
            text_part=self.text_part,
        )
        optimizer = self.train_config.optimizer(
            clip.parameters(), **self.train_config.optimizer_params
        )
        if self.train_config.scheduler is not None:
            scheduler = self.train_config.scheduler(
                optimizer, **self.train_config.scheduler_params
            )
        else:
            scheduler = None
        return Engine(
            seed=self.train_config.seed,
            clip=clip,
            criterion_clip=self.train_config.criterion_clip,
            criterion_image=self.train_config.criterion_image,
            criterion_text=self.train_config.criterion_text,
            optimizer=optimizer,
            device=self.train_config.device,
            scheduler=scheduler,
            count_accumulated_batches=self.train_config.accumulation_steps,
        )

    def configurate(self) -> TrainingParameters:
        loaders = self.configurate_dataloaders()
        engine = self.configurate_engine()
        save_dir = Path(self.train_config.save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        return TrainingParameters(
            n_epochs=self.train_config.n_epochs,
            engine=engine,
            dataloaders=loaders,
            save_dir=save_dir,
            coefficients=self.train_config.coefficients,
        )
