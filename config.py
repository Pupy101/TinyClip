from typing import Any, Dict, Union

from torch import optim, nn
from transformers import BertTokenizer


class Config:
    TYPE_USING: str = 'train'  # or 'eval'

    DATASETS_CSV = {
        'train': '/content/train.csv',
        'valid': '/content/valid.csv'
    }
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_SEQUENCE_LEN = 20

    LOADER_PARAMS: Dict[str, Dict[str, Union[bool, int]]] = {
        'train': {
            'batch_size': 384,
            'shuffle': True
        },
        'valid': {
            'batch_size': 384,
            'shuffle': False
        }
    }

    MODEL_IMAGE_NAME: str = 'wide_resnet50_2'
    MODEL_IMAGE_PARAMETERS: Dict[str, Any] = {
        'pretrained': True
    }

    MODEL_TEXT_NAME: str = 'BertForSequenceClassification'
    MODEL_TEXT_PARAMETERS: Dict[str, Any] = {
        'pretrained': True,
        'name_pretrained': 'bert-base-uncased'
    }

    PATH_TO_WEIGHTS: Dict[str, Union[str, None]] = {
        'PRETRAINED_WEIGHTS': None,
        'PATH_TO_SAVE': './train/weights'
    }

    NUM_EPOCH: int = 30
    ACCUMULATE: bool = True
    ACCUMULATION_STEPS: int = 2

    OPTIMIZER: nn.Module = optim.AdamW
    OPTIMIZER_PARAMS: dict = {
        'lr': 1e-3
    }

    CRITERION: nn.Module = nn.CrossEntropyLoss

    SCHEDULER_LR = optim.lr_scheduler.OneCycleLR
    SCHEDULER_LR_PARAMS = {
        'anneal_strategy': 'cos'
    }

    INFERENCE_PARAMS: Dict[str, Any] = {
        'TARGET_DIR': '/content/CLIP/test',
        'IMAGES_DIR': '/content/CLIP/predict',
        'CLASSES': ['Dog', 'Cat', 'Human', 'Car']
    }
