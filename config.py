from typing import Dict, Union

from torch import optim, nn
from model import clip


class Config:
    DATASET_PATH: Dict[str, Union[None, str]] = {
        'train': './train_path'
    }

    LOADER_PARAMS: Dict[str, Dict[str, Union[bool, int]]] = {
        'train': {
            'batch_size': 16,
            'shuffle': True
        },
        'valid': {
            'batch_size': 32,
            'shuffle': False
        }
    }

    MODEL = clip

    IND_REQUIRES_GRAD_IMAGE_NET = -40 requires_grad
    IND_REQUIRES_GRAD_TEXT_NET = -40

    OPTIMIZER = optim.AdamW
    OPTIMIZER_PARAMS = {
        'params': [
            *list(MODEL.model_img_emb.parameters())[:-30],
            *list(MODEL.model_text_emb.parameters())[:-30]
        ],
        'lr': 3e-4,
    }