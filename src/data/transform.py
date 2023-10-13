from dataclasses import dataclass
from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2
from git import Sequence

from src.types import DatasetMark


@dataclass
class ImageTransform:
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    crop: int = 250
    size: int = 224

    def __post_init__(self) -> None:
        assert len(self.mean) == 3
        assert len(self.std) == 3

    def create(self, transform_type: str) -> A.Compose:
        mark = DatasetMark(transform_type)

        augmentations: List[A.BasicTransform] = []

        if mark is DatasetMark.TRAIN:
            augmentations.extend(
                [
                    A.SmallestMaxSize(max_size=self.crop),
                    A.RandomCrop(height=self.size, width=self.size),
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
            augmentations.extend(
                [A.SmallestMaxSize(max_size=self.crop), A.CenterCrop(height=self.size, width=self.size)]
            )
        else:
            raise ValueError(f"Strange transform_type for 'create_image_augmentations': {mark}")

        augmentations.extend([A.Normalize(mean=self.mean, std=self.std), ToTensorV2()])
        return A.Compose(augmentations)
