import albumentations as A

from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=224),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(),
        A.CoarseDropout(),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.Rotate(limit=15),
        A.Normalize(),
        ToTensorV2()
    ]
)

valid_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=224),
        A.CenterCrop(height=224, width=224),
        A.Normalize(),
        ToTensorV2()
    ]
)
