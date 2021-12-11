from typing import Any, Dict, Union, Callable
from collections import OrderedDict

from torch import optim, nn
from transformers import (
    DistilBertTokenizer,
    BertTokenizer
)

from utils import augmentations


class Config:
    TYPE_USING: str = 'train'  # or 'eval'

    # for training i use
    # https://www.kaggle.com/mrviswamitrakaushik/image-captioning-data
    # otherwise you can use csv if change 'json' to 'csv'
    DATASET_PARAMS: Dict[str, Union[str, int, Callable]] = {
        'jsons': [
            '/content/caption_datasets/dataset_coco.json',
            '/content/caption_datasets/dataset_flickr30k.json',
            '/content/caption_datasets/dataset_flickr8k.json'
        ],
        'dir_image': '/content/train2014/train2014',
        'tokenizer': BertTokenizer.from_pretrained(
            'bert-base-uncased'
        ),
        'max_size_seq_len': 30,
        'transforms': {
            'train': augmentations.train,
            'valid': augmentations.valid
        }
    }

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

    MODEL_IMAGE_NAME: str = 'wide_resnet50'
    MODEL_IMAGE_PARAMETERS: Dict[str, Any] = {
        'pretrained': True
    }

    MODEL_TEXT_NAME: str = 'BertForSequenceClassification'
    MODEL_TEXT_PARAMETERS: Dict[str, Any] = {
        'pretrained': True,
        'name_pretrained': 'bert-base-uncased'
    }

    PATH_TO_WEIGHTS: Union[str, None] = None

    NUM_EPOCH: int = 30
    ACCUMULATE: bool = True

    OPTIMIZER: nn.Module = optim.AdamW
    OPTIMIZER_PARAMS: dict = {
        'lr': 1e-3
    }

    CRITERION: nn.Module = nn.CrossEntropyLoss

    SCHEDULER_LR = optim.lr_scheduler.OneCycleLR
    SCHEDULER_LR_PARAMS = {
        'anneal_strategy': 'cos'
    }

    PATH_TO_SAVE_MODEL_WEIGHTS: str = './train_result'

    INFERENCE_PARAMS: Dict[str, Any] = {
        'TARGET_DIR': '/content/CLIP/test',
        'IMAGES_DIR': '/content/CLIP/predict',
        'CLASSES': [
            'Dog',
            'Cat',
            'Human',
            'Car'
        ]
    }
