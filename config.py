from typing import Dict, Union
from collections import OrderedDict

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

    OPTIMIZER = optim.AdamW

    CRITERION = nn.CrossEntropyLoss()

    TRAINING_STAGES = OrderedDict(
        {
            'Stage 1': {
                'lr': 5e-5,
                'params': [
                    *list(MODEL.clf_img.parameters()),
                    *list(MODEL.clf_text.parameters())
                    ],
                'freeze': {
                    'model_img_emb': {
                        'model': MODEL.model_img_emb,
                        'last_index_freeze': 0,
                        'freeze_all_net': True
                    },
                    'model_text_emb': {
                        'model': MODEL.model_text_emb,
                        'last_index_freeze': 0,
                        'freeze_all_net': True
                    }
                },
                'n_epoch': 10
            },
            'Stage 2': {
                'lr': 1e-4,
                'params': [
                    *list(MODEL.clf_img.parameters()),
                    *list(MODEL.model_img_emb.parameters())[-30:],
                    *list(MODEL.clf_text.parameters()),
                    *list(MODEL.model_text_emb.parameters())[-30:],
                    ],
                'unfreeze': {
                    'model_img_emb': {
                        'model': MODEL.model_img_emb,
                        'first_index_unfreeze': -30
                    },
                    'model_text_emb': {
                        'model': MODEL.model_text_emb,
                        'first_index_unfreeze': -30
                    }
                },
                'n_epoch': 20
            },
            'Stage 3': {
                'lr': 3e-4,
                'params': [
                    *list(MODEL.clf_img.parameters()),
                    *list(MODEL.model_img_emb.parameters()),
                    *list(MODEL.clf_text.parameters()),
                    *list(MODEL.model_text_emb.parameters()),
                    ],
                'unfreeze': {
                    'model_img_emb': {
                        'model': MODEL.model_img_emb
                    },
                    'model_text_emb': {
                        'model': MODEL.model_text_emb
                    }
                },
                'n_epoch': 40
            }
        }
    )
