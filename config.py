from typing import Any, Dict, Union

from torch import optim, nn
from transformers import AutoTokenizer

from utils import FocalLoss


class Config:
    TYPE_USING: str = 'train'  # or 'eval'

    DATASETS_CSV = {
        'train': '/content/train.csv',
        'valid': '/content/valid.csv'
    }
    TOKENIZER = AutoTokenizer.from_pretrained('cointegrated/LaBSE-en-ru')
    MAX_SEQUENCE_LEN = 20

    LOADER_PARAMS: Dict[str, Dict[str, Union[bool, int]]] = {
        'train': {
            'batch_size': 840,
            'shuffle': True
        },
        'valid': {
            'batch_size': 840,
            'shuffle': True
        }
    }

    MODEL_IMAGE_NAME: str = 'mobilenet_v3_small'
    MODEL_IMAGE_PARAMETERS: Dict[str, Any] = {
        'pretrained': True
    }

    MODEL_TEXT_NAME: str = 'AutoModel'
    MODEL_TEXT_PARAMETERS: Dict[str, Any] = {
        'pretrained': True,
        'name_pretrained': 'cointegrated/LaBSE-en-ru'
    }

    PATH_TO_WEIGHTS: Dict[str, Union[str, None]] = {
        'PRETRAINED_WEIGHTS': None,
        'PATH_TO_SAVE': './training/weights'
    }

    NUM_EPOCH: int = 30
    ACCUMULATE: bool = True
    ACCUMULATION_STEPS: int = 2

    OPTIMIZER: nn.Module = optim.AdamW
    OPTIMIZER_PARAMS: dict = {
        'lr': 1e-3
    }

    CRITERION: nn.Module = FocalLoss

    SCHEDULER_LR = optim.lr_scheduler.OneCycleLR
    SCHEDULER_LR_PARAMS = {
        'anneal_strategy': 'cos'
    }

    INFERENCE_PARAMS: Dict[str, Any] = {
        'TARGET_DIR': '/content/CLIP/test',
        'IMAGES_DIR': '/content/CLIP/predict',
        'CLASSES': ['Dog', 'Cat', 'Human', 'Car']
    }
