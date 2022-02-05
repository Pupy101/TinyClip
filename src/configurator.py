import os

from typing import Any, Dict, Union

import torch
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from config import Config
from .model import (
    CLIP,
    VisionPartCLIP,
)
from .utils.augmentations import augmentations
from .utils.dataset import (
    ImageFromCSV,
    TextAndImageFromCSV,
    TextAndImageCachedTextFromCSV,
)
from .utils.misc import freeze_weight


class Configurator:
    """
    Class for configuration model and other
    parameters for train or eval mode
    """

    def __init__(self, config: Config) -> None:
        """
        Method for init training parameters

        Args:
            config: config of training
        """
        type_using = config.TYPE_USING.lower().strip()
        assert type_using in ['train', 'eval'], 'Incorrect type of using'
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.training = type_using == 'train'
        self.model = self._init_model()
        self.loaders = self._init_loaders()
        if config.TYPE_USING == 'train':
            self.optimizer = self._init_optimizer_and_freeze()
            self.scheduler = self._init_scheduler()
            self.criterion = self._init_criterion()

    @property
    def train_parameters(self) -> Dict[str, Any]:
        """
        Method return training parameters

        Returns:
            dict with training parameters
        """
        assert self.training, 'Use only for training'
        assert self.config.ACCUMULATION > 0, 'Accumulation must be more than 0'
        return {
            'accumulation': self.config.ACCUMULATION,
            'criterion': self.criterion,
            'device': self.device,
            'loaders': self.loaders,
            'model': self.model,
            'n_epoch': self.config.NUM_EPOCH,
            'optimizer': self.optimizer,
            'save_dir': self.config.PATH_TO_WEIGHT['SAVING'],
            'scheduler': self.scheduler,
        }

    @property
    def eval_parameters(self) -> Dict[str, Any]:
        """
        Method return evaluation parameters

        Returns:
            dict with evaluation parameters
        """
        assert not self.training, 'Use only for evaluation'
        inference_params = self.config.INFERENCE_PARAMS
        return {
            'classes': inference_params['CLASSES'],
            'csv': self.csv,
            'device': self.device,
            'loaders': self.loaders,
            'model': self.model,
            'target_dir': inference_params['PREDICTION_DIR'],
            'tokenizer': self.config.TOKENIZER,
        }

    def _init_model(self) -> nn.Module:
        """
        Method for init model

        Returns:
            CLIP model
        """
        vision_part = VisionPartCLIP(self.config.MODEL_VISION)
        model = CLIP(vision_part, self.config.MODEL_TEXT)
        if self.config.PATH_TO_WEIGHT['PRETRAINED']:
            weights = torch.load(
                self.config.PATH_TO_WEIGHT['PRETRAINED']
            )
            model.load_state_dict(weights)
        return model.to(self.device)

    def _init_criterion(self) -> nn.Module:
        """
        Method for create criterion

        Returns:
            criterion
        """
        assert self.training, 'Init only in training mode'
        return self.config.CRITERION()

    def _init_optimizer_and_freeze(self) -> optim.Optimizer:
        """
        Method freeze weight of text part CLIP and create optimizer
        :return: optimizer
        """
        assert self.training, 'Init only in training mode'
        assert hasattr(self, 'model'), 'Please init model before'
        freeze_weight(self.model.text_model)
        return self.config.OPTIMIZER(
            self.model.vision_part.parameters(),
            **self.config.OPTIMIZER_PARAMS
        )

    def _init_dataset(self) -> Dict[str, Dataset]:
        """
        Method create dataset

        Returns:
            dict with datasets
        """
        datasets = {}
        if self.training:
            datasets_csv = self.config.DATASETS_CSV
            for key in datasets_csv:
                transformation = (
                    augmentations[key]
                    if key in augmentations
                    else augmentations['valid']
                )
                dataset_initializer = (
                    TextAndImageCachedTextFromCSV
                    if self.config.DATASET_WITH_CACHED_TEXT
                    else TextAndImageFromCSV
                )
                datasets[key] = dataset_initializer(
                    csv=pd.read_csv(datasets_csv[key]),
                    tokenizer=self.config.TOKENIZER,
                    max_seq_len=self.config.MAX_SEQUENCE_LEN,
                    transform=transformation,
                )
        else:
            eval_dirs = self.config.INFERENCE_PARAMS
            csv = pd.DataFrame({
                'image': [
                    os.path.join(eval_dirs['IMAGES_DIR'], img)
                    for img in os.listdir(eval_dirs['IMAGES_DIR'])
                ]
            })
            self.csv = csv
            datasets['valid'] = ImageFromCSV(
                csv=csv, transform=augmentations['valid']
            )
        return datasets

    def _init_loaders(self) -> Dict[str, DataLoader]:
        """
        Method for init loaders

        Returns:
            dict with loaders
        """
        datasets = self._init_dataset()
        return {
            key: DataLoader(
                datasets[key],
                **self.config.LOADER_PARAMS[key]
            )
            for key in datasets
        }

    def _init_scheduler(self) -> Union[None, optim.lr_scheduler._LRScheduler]:
        """
        Method for init learning rate scheduler

        Returns:
            None or lr scheduler
        """
        assert self.training, 'Init only in training mode'
        assert hasattr(self, 'optimizer'), 'Please init optimizer before'
        assert hasattr(self, 'loaders'), 'Please init loaders before'
        if not self.config.SCHEDULER_LR:
            return None
        len_loader = (
                len(self.loaders['train']) //
                self.config.ACCUMULATION + 1
        )
        return self.config.SCHEDULER_LR(
            self.optimizer,
            max_lr=[param['lr'] for param in self.optimizer.param_groups],
            epochs=self.config.NUM_EPOCH,
            steps_per_epoch=len_loader,
            **self.config.SCHEDULER_LR_PARAMS
        )
