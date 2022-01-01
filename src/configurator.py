import os
from typing import Any, Dict

import torch
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import config

from .clip import (
    configuration_image_model,
    configuration_text_model,
    CLIP
)
from utils import (
    augmentations,
    freeze_weight,
    ImageFromCSV,
    TextAndImageFromCSV
)


class Configurator:
    """
    Class for configuration model and other
    parameters for train or eval mode
    """

    def __init__(self, config: config.Config):
        """
        Method for init training parameters
        :param config: config of training
        """
        self.config = config
        self.training = config.TYPE_USING == 'train'
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
        :return: dict with parameters
        """
        assert self.training, 'Use only for training'
        return {
            'config': self.config,
            'criterion': self.criterion,
            'loaders': self.loaders,
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }

    @property
    def eval_parameters(self) -> Dict[str, Any]:
        """
        Method return evaluation parameters
        :return: dict with parameters
        """
        assert not self.training, 'Use only for evaluation'
        INFERENCE_PARAMS = self.config.INFERENCE_PARAMS
        return {
            'classes': INFERENCE_PARAMS['CLASSES'],
            'config': self.config,
            'csv': self._csv,
            'loaders': self.loaders,
            'model': self.model,
            'target_dir': INFERENCE_PARAMS['TARGET_DIR'],
            'tokenizer': self.config.TOKENIZER
        }

    def _init_model(self) -> nn.Module:
        """
        Method for init model
        :return: CLIP
        """
        image_model, image_model_shape = configuration_image_model(
            self.config.MODEL_IMAGE_NAME,
            **self.config.MODEL_IMAGE_PARAMETERS
        )
        text_model, text_model_shape = configuration_text_model(
            self.config.MODEL_TEXT_NAME,
            **self.config.MODEL_TEXT_PARAMETERS
        )
        model = CLIP(
            image_model, image_model_shape,
            text_model, text_model_shape
        )
        if self.config.PATH_TO_WEIGHTS['PRETRAINED_WEIGHTS']:
            weights = torch.load(
                self.config.PATH_TO_WEIGHTS['PRETRAINED_WEIGHTS']
            )
            model.load_state_dict(weights)
        return model
    
    def _init_criterion(self) -> nn.Module:
        """
        Method for create criterion
        :return: criterion
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
        freeze_weight(self.model.model_text_embedding)
        return self.config.OPTIMIZER(
            self.model.image_model.parameters(),
            **self.config.OPTIMIZER_PARAMS
        )

    def _init_dataset(self) -> Dict[str, Dataset]:
        """
        Method create dataset
        :return: dict with datasets
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
                datasets[key] = TextAndImageFromCSV(
                    csv=pd.read_csv(datasets_csv[key]),
                    tokenizer=self.config.TOKENIZER,
                    max_seq_len=self.config.MAX_SEQUENCE_LEN,
                    transform=transformation
                )
        elif self.config.TYPE_USING == 'eval':
            EVAL_DIRS = self.config.INFERENCE_PARAMS
            csv = pd.DataFrame({
                'image': [
                    os.path.join(EVAL_DIRS['IMAGES_DIR'], img)
                    for img in os.listdir(EVAL_DIRS['IMAGES_DIR'])
                ]
            })
            self._csv = csv
            datasets['valid'] = ImageFromCSV(
                csv=csv, transform=augmentations['valid']
            )
        return datasets

    def _init_loaders(self):
        """
        Method for init loaders
        :return: dict with loaders
        """
        datasets = self._init_dataset()
        return {
            key: DataLoader(
                datasets[key],
                **self.config.LOADER_PARAMS[key]
            )
            for key in datasets
        }

    def _init_scheduler(self):
        assert self.training, 'Init only in training mode'
        assert hasattr(self, 'optimizer'), 'Please init optimizer before'
        assert hasattr(self, 'loaders'), 'Please init loaders before'
        if not self.config.SCHEDULER_LR:
            return None
        len_loader = (
            len(self.loaders['train']) // self.config.ACCUMULATION_STEPS + 1
            if self.config.ACCUMULATE 
            else len(self.loaders['train']) + 1
        )
        return self.config.SCHEDULER_LR(
            self.optimizer,
            max_lr=[param['lr'] for param in self.optimizer.param_groups],
            epochs=self.config.NUM_EPOCH,
            steps_per_epoch=len_loader,
            **self.config.SCHEDULER_LR_PARAMS
        )
