from typing import Dict, Union

from torch import optim, nn
from transformers import DistilBertTokenizer

from model import clip
from utils import utils, augmentations



class Config:
    TYPE_USING = 'train'  # or 'eval'

    DATASET_PARAMS: Dict[str, Union[str, int]] = {
        'path_to_csv': '/content/flickr30k_images/results.csv',
        'dir_image': '/content/flickr30k_images/flickr30k_images',
        'tokenizer': DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        'max_size_seq_len': 70,
        'transform': {
            'train': augmentations.train_transform,
            'valid': augmentations.valid_transform
        }
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

    PATH_TO_SAVE_MODEL_WEIGHTS = './train_result'

    IND_REQUIRES_GRAD_IMAGE_NET: Union[int, None] = 0
    IND_REQUIRES_GRAD_TEXT_NET: Union[int, None] = 0

    if IND_REQUIRES_GRAD_IMAGE_NET is not None:
        utils.freeze_weights(clip.model_img_emb, IND_REQUIRES_GRAD_IMAGE_NET)
    if IND_REQUIRES_GRAD_TEXT_NET is not None:
        utils.freeze_weights(clip.model_text_emb, IND_REQUIRES_GRAD_TEXT_NET)

    OPTIMIZER = optim.AdamW
    OPTIMIZER_PARAMS = {
        'params': [
            *list(MODEL.model_img_emb.parameters())[IND_REQUIRES_GRAD_IMAGE_NET:],
            *list(MODEL.model_text_emb.parameters())[IND_REQUIRES_GRAD_TEXT_NET:]
        ],
        'lr': 3e-4,
    }

    CRITERION = nn.CrossEntropyLoss()

    N_EPOCH = 10
