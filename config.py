from typing import Dict, Union
from collections import OrderedDict

from torch import optim, nn
from transformers import DistilBertTokenizer

from model import CLIP
from utils import augmentations



class Config:
    TYPE_USING = 'train'  # or 'eval'

    DATASET_PARAMS: Dict[str, Union[str, int]] = {
        'jsons': [
            '/content/caption_datasets/dataset_coco.json',
            '/content/caption_datasets/dataset_flickr30k.json',
            '/content/caption_datasets/dataset_flickr8k.json'
            ],
        'dir_image': '/content/train2014/train2014',
        'tokenizer': DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        'max_size_seq_len': 70,
        'transform': {
            'train': augmentations.train_transform,
            'valid': augmentations.valid_transform
        }
    }

    LOADER_PARAMS: Dict[str, Dict[str, Union[bool, int]]] = {
        'train': {
            'batch_size': 192,
            'shuffle': True
        },
        'valid': {
            'batch_size': 192,
            'shuffle': False
        }
    }

    MODEL = CLIP('wide_resnet50', 'distilbert', 1024)

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
                'n_epoch': 5
            },
            'Stage 2': {
                'lr': 6e-5,
                'params': [
                    *list(MODEL.clf_img.parameters()),
                    *list(MODEL.model_img_emb.parameters())[-200:],
                    *list(MODEL.clf_text.parameters()),
                    *list(MODEL.model_text_emb.parameters())[-200:],
                    ],
                'unfreeze': {
                    'model_img_emb': {
                        'model': MODEL.model_img_emb,
                        'first_index_unfreeze': -200
                    },
                    'model_text_emb': {
                        'model': MODEL.model_text_emb,
                        'first_index_unfreeze': -200
                    }
                },
                'n_epoch': 10
            },
            'Stage 3': {
                'lr': 2e-5,
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
                'n_epoch': 15
            }
        }
    )
