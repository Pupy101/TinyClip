from typing import Callable, Dict, List, Union

from torch import optim, nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

from src.utils import FocalLoss
from src.model import WrapperModelFromHuggingFace


class Config:
    TYPE_USING: str = 'train'  # or 'eval'

    DATASETS_CSV: Dict[str, str] = {
        'train': '/content/train.csv',
        'valid': '/content/valid.csv',
    }
    DATASET_WITH_CACHED_TEXT: bool = True

    LOADER_PARAMS: Dict[str, Dict[str, Union[bool, int]]] = {
        'train': {
            'batch_size': 840,
            'shuffle': True,
        },
        'valid': {
            'batch_size': 840,
            'shuffle': True,
        }
    }

    MODEL_VISION: nn.Module = models.mobilenet_v3_small(pretrained=True)
    # change last part of pretrained on imagenet model
    MODEL_VISION.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=2048),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(in_features=2048, out_features=768),
        nn.LayerNorm(normalized_shape=768),
    )

    MAX_SEQUENCE_LEN: int = 20
    TOKENIZER: Callable = AutoTokenizer.from_pretrained('cointegrated/LaBSE-en-ru')

    MODEL_TEXT: nn.Module = WrapperModelFromHuggingFace(
        AutoModel.from_pretrained('cointegrated/LaBSE-en-ru'),
    )

    PATH_TO_WEIGHT: Dict[str, Union[None, str]] = {
        'PRETRAINED': None,
        'SAVING': './training/weights',
    }

    DEVICE: str = 'cuda'
    NUM_EPOCH: int = 30
    ACCUMULATION: int = 2  # set 1 if accumulation doesn't need

    OPTIMIZER: optim.Optimizer = optim.AdamW
    OPTIMIZER_PARAMS: Dict[str, float] = {'lr': 3e-4}

    CRITERION: nn.Module = FocalLoss
    CRITERION_PARAMS: Dict[str, Union[int, float]] = {
        'alpha': 0.2, 'gamma': 2,
    }

    SCHEDULER_LR = optim.lr_scheduler.OneCycleLR  # other scheduler or None
    SCHEDULER_LR_PARAMS = {'anneal_strategy': 'cos'}

    INFERENCE_PARAMS: Dict[str, Union[str, List[str]]] = {
        'PREDICTION_DIR': '/content/CLIP/prediction',
        'IMAGES_DIR': '/content/CLIP/testing',
        'CLASSES': ['Dog', 'Cat', 'Human', 'Car'],
    }
