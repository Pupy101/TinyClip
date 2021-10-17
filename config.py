from typing import Dict, Union

from torch import optim, nn
from model import clip

from utils import utils


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

    IND_REQUIRES_GRAD_IMAGE_NET: Union[int, None] = -40
    IND_REQUIRES_GRAD_TEXT_NET: Union[int, None] = -40

    if IND_REQUIRES_GRAD_IMAGE_NET is not None:
        utils.freeze_weights(clip.model_img_emb, IND_REQUIRES_GRAD_IMAGE_NET)
    if IND_REQUIRES_GRAD_TEXT_NET is not None:
        utils.freeze_weights(clip.model_text_emb, IND_REQUIRES_GRAD_TEXT_NET)

    OPTIMIZER = optim.AdamW
    OPTIMIZER_PARAMS = {
        'params': [
            *list(MODEL.model_img_emb.parameters())[:-30],
            *list(MODEL.model_text_emb.parameters())[:-30]
        ],
        'lr': 3e-4,
    }