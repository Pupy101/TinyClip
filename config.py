from pathlib import Path
from typing import Callable, Dict, List, Union

import torch

from torch import optim, nn
from transformers import AutoTokenizer, AutoModel

from src.utils.losses import FocalLoss
from src.model import WrapperModelFromHuggingFace, VisionModelPreparator
from src.classes.config import (
    DatasetType,
    CLIPDatasets,
    CLIPLoaders,
    InferenceParameters,
    LoaderParameters,
    ModelWeight,
    TypeUsing,
)


class Config:
    TYPE_USING: TypeUsing = TypeUsing.TRAIN

    DATASET_TYPE: DatasetType = DatasetType.URL
    DATASETS_CSV: CLIPDatasets = CLIPDatasets(
        train=Path('/content/train.csv'),
        valid=Path('/content/valid.csv'),
    )

    LOADER_PARAMS: CLIPLoaders = CLIPLoaders(
        train=LoaderParameters(
            batch_size=840,
            shuffle=True,
        ),
        valid=LoaderParameters(
            batch_size=840,
            shuffle=True,
        ),
    )

    MODEL_VISION: nn.Module = VisionModelPreparator(model='mobilenet_v3_small', pretrained=True)\
        .change_layer_to_mlp(layer_name='classifier', mlp_shapes=[576, 640, 768], activation=nn.PReLU).model

    MAX_SEQUENCE_LEN: int = 24
    TOKENIZER: Callable = AutoTokenizer.from_pretrained('cointegrated/LaBSE-en-ru')

    MODEL_TEXT: nn.Module = WrapperModelFromHuggingFace(
        AutoModel.from_pretrained('cointegrated/LaBSE-en-ru'),
        getting_attr_from_model='pooler_output',
    )

    PATH_TO_WEIGHT: ModelWeight = ModelWeight(
        save=Path('./training/weights'), pretrained=None
    )

    DEVICE: torch.device = torch.device('cuda')
    NUM_EPOCH: int = 30
    ACCUMULATION: int = 16

    OPTIMIZER: optim.Optimizer = optim.AdamW
    OPTIMIZER_PARAMS: Dict[str, float] = {'lr': 3e-4}

    CRITERION: nn.Module = FocalLoss
    CRITERION_PARAMS: Dict[str, Union[int, float]] = {
        'alpha': 0.2, 'gamma': 2,
    }

    SCHEDULER_LR = optim.lr_scheduler.OneCycleLR  # other scheduler or None
    SCHEDULER_LR_PARAMS = {'anneal_strategy': 'cos'}

    INFERENCE_PARAMS: InferenceParameters = InferenceParameters(
        image_dir=Path('/content/CLIP/testing'),
        prediction_dir=Path('/content/CLIP/prediction'),
        classes=['Dog', 'Cat', 'Human', 'Car'],
    )
