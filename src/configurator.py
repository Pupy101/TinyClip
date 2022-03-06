from dataclasses import asdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, Type, Union, TYPE_CHECKING

import torch
import pandas as pd

from requests import Session
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from .classes.configurator import TrainParameters, EvaluationParameters
from .data.augmentations import augmentations
from .data.dataset import (
    InferenceImage, TextAndImage, TextAndImageCachedText, TextAndImageURL
)
from .data.collate_funcs import url_collate
from .model import CLIP, VisionPartCLIP
from .utils.functions import freeze_weight


if TYPE_CHECKING:
    from config import Config


class Configurator:
    """
    Class for configuration model and other
    parameters for train or eval mode
    """

    def __init__(self, path_to_config: Union[str, Path]):
        """
        Method for init training parameters

        Args:
            path_to_config: path to clip config
        """
        if isinstance(path_to_config, str):
            path_to_config = Path(path_to_config)
        assert path_to_config.exists(), 'Fix config path'
        module_with_config = SourceFileLoader(
            path_to_config.stem, str(path_to_config.absolute())
        ).load_module()
        assert 'Config' in dir(module_with_config), 'Module must consist class Config'
        config: 'Config' = module_with_config.Config
        type_using = config.TYPE_USING.value
        assert type_using in ['train', 'evaluation'], 'Incorrect type of using'
        self.config = config
        self.device = config.DEVICE
        self.training = type_using == 'train'
        self.model = self._init_model()
        if self.training and self.config.DATASET_TYPE.value == 'url':
            self.session = Session()
        self.loaders = self._init_loaders()
        if self.training:
            self.optimizer = self._init_optimizer_and_freeze()
            self.scheduler = self._init_scheduler()
            self.criterion = self._init_criterion()

    @property
    def train_parameters(self) -> TrainParameters:
        """
        Method return training parameters

        Returns:
            dict with training parameters
        """
        assert self.training, 'Use only for training'
        assert self.config.ACCUMULATION > 0, 'Accumulation must be more than 0'
        return TrainParameters(
            accumulation=self.config.ACCUMULATION,
            criterion=self.criterion,
            device=self.device,
            loaders=self.loaders,
            model=self.model,
            n_epoch=self.config.NUM_EPOCH,
            optimizer=self.optimizer,
            save_dir=self.config.PATH_TO_WEIGHT.save,
            scheduler=self.scheduler,
        )

    @property
    def evaluation_parameters(self) -> EvaluationParameters:
        """
        Method return evaluation parameters

        Returns:
            dict with evaluation parameters
        """
        assert not self.training, 'Use only for evaluation'
        return EvaluationParameters(
            classes=self.config.INFERENCE_PARAMS.classes,
            csv=self.csv,
            device=self.device,
            loaders=self.loaders,
            model=self.model,
            target_dir=self.config.INFERENCE_PARAMS.prediction_dir,
            tokenizer=self.config.TOKENIZER,
        )

    def _init_model(self) -> CLIP:
        """
        Method for init model

        Returns:
            CLIP model
        """
        vision_part = VisionPartCLIP(self.config.MODEL_VISION)
        model = CLIP(vision_part, self.config.MODEL_TEXT)
        if self.config.PATH_TO_WEIGHT.pretrained:
            weights = torch.load(
                self.config.PATH_TO_WEIGHT.pretrained
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
            self.model.cv_model.parameters(),
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
            datasets_csv = asdict(self.config.DATASETS_CSV)
            for key in datasets_csv:
                transformation = (
                    augmentations[key]
                    if key in augmentations
                    else augmentations['valid']
                )
                if self.config.DATASET_TYPE.value == 'cached':
                    datasets[key] = TextAndImageCachedText(
                        csv=pd.read_csv(datasets_csv[key]),
                        transform=transformation,
                    )
                elif self.config.DATASET_TYPE.value == 'url':
                    datasets[key] = TextAndImageURL(
                        csv=pd.read_csv(datasets_csv[key]),
                        tokenizer=self.config.TOKENIZER,
                        max_seq_len=self.config.MAX_SEQUENCE_LEN,
                        transform=transformation,
                        session=self.session,
                    )
                else:
                    datasets[key] = TextAndImage(
                        csv=pd.read_csv(datasets_csv[key]),
                        tokenizer=self.config.TOKENIZER,
                        max_seq_len=self.config.MAX_SEQUENCE_LEN,
                        transform=transformation,
                    )
        else:
            csv = pd.DataFrame({
                'image': list(self.config.INFERENCE_PARAMS.image_dir.iterdir())
            })
            self.csv = csv
            datasets['valid'] = InferenceImage(
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
        loader_parameters = asdict(self.config.LOADER_PARAMS)
        if self.config.DATASET_TYPE.value == 'url':
            for key in loader_parameters:
                loader_parameters[key].update(
                    {'collate_fn': url_collate}
                )
        return {
            key: DataLoader(
                datasets[key],
                **loader_parameters[key]
            )
            for key in datasets
        }

    def _init_scheduler(self) -> Union[None, Type[optim.lr_scheduler.ReduceLROnPlateau]]:
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
