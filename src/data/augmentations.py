import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.types import Augmentations

# augmentations for train
train = A.Compose(
    [
        A.SmallestMaxSize(max_size=230),
        A.RandomCrop(height=224, width=224),
        A.Flip(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(),
        A.GaussNoise(),
        A.CoarseDropout(),
        A.ChannelDropout(),
        A.Rotate(limit=15),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# augmentations for validation
validation = A.Compose(
    [
        A.SmallestMaxSize(max_size=230),
        A.CenterCrop(height=224, width=224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

augmentations = Augmentations(train=train, validation=validation)
