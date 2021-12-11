import torch

from torch.utils.data import DataLoader

from .clip import (
    configuration_image_model,
    configuration_text_model,
    CLIP
)
from utils import freeze_weight
from utils.dataset import (
    create_datasets_from_json,
    create_datasets_from_csv
)


class Configurator:
    """
    Class for configuration model and other
    parameters for train or eval mode
    """

    def __init__(self, config):
        self.config = config

    def init_all(self):
        self.model = self._init_model()
        self.optimizer = self._init_optimizer_and_freeze()
        self.loaders = self._init_loaders()
        self.scheduler = self._init_scheduler()
        self.criterion = self._init_criterion()
        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'criterion': self.criterion,
            'loaders': self.loaders
        }


    def _init_model(self):
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
        if self.config.PATH_TO_WEIGHTS:
            weights = torch.load(self.config.PATH_TO_WEIGHTS)
            model.load_state_dict(weights)
        return model
    
    def _init_criterion(self):
        return self.config.CRITERION()

    def _init_optimizer_and_freeze(self):
        assert hasattr(self, 'model'), 'Please init model before'
        freeze_weight(self.model.model_text_embedding)
        return self.config.OPTIMIZER(
            self.model.parameters(),
            **self.config.OPTIMIZER_PARAMS
        )

    def _init_dataset(self):
        if 'jsons' in self.config.DATASET_PARAMS:
            dataset = create_datasets_from_json(
                **self.config.DATASET_PARAMS
            )
        elif 'csv':
            dataset = create_datasets_from_csv(
                **self.config.DATASET_PARAMS
            )
        else:
            raise ValueError('Set right params in config DATASET_PARAMS')
        return dataset

    def _init_loaders(self):
        dataset = self._init_dataset()
        return {
            key: DataLoader(
                dataset[key]
                **self.config.LOADER_PARAMS[key]
            )
            for key in dataset
        }

    def _init_scheduler(self):
        assert hasattr(self, 'optimizer'), 'Please init optimizer before'
        assert hasattr(self, 'loaders'), 'Please init loaders before'
        if not self.config.SCHEDULER_LR:
            return None
        len_loader = (
            len(self.loaders['train']) // 2 + 1
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
