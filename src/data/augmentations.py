from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.types import DatasetMark


def create_augmentations(transform_type: str) -> A.Compose:
    mark = DatasetMark(transform_type)

    augmentations: List[A.BasicTransform] = []

    if mark is DatasetMark.TRAIN:
        augmentations.extend(
            [
                A.SmallestMaxSize(max_size=250),
                A.RandomCrop(height=224, width=224),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.CLAHE(),
                A.GaussNoise(),
                A.CoarseDropout(),
                A.ChannelDropout(),
                A.Rotate(limit=15),
            ]
        )
    elif mark is DatasetMark.VAL or mark is DatasetMark.TEST:
        augmentations.extend([A.SmallestMaxSize(max_size=250), A.CenterCrop(height=224, width=224)])
    else:
        raise ValueError(f"Strange transform_type for 'create_image_augmentations': {mark}")

    augmentations.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(augmentations)
