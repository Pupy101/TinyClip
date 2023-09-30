from typing import Tuple, Union

import albumentations as A
import numpy as np
from pandas import DataFrame
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from clip.data.augmentations import ComposeAugmentator


class ImageDataset(Dataset):
    def __init__(
        self,
        dataframe: DataFrame,
        image_transform: A.Compose,
        image_column: str = "image",
        text_column: str = "text",
    ) -> None:
        assert image_column in list(dataframe.columns), "Write right name for image column"
        assert text_column in list(dataframe.columns), "Write right name for label column"
        self.df = dataframe
        self.img_col_idx = list(self.df.columns).index(image_column)
        self.text_col_idx = list(self.df.columns).index(text_column)
        self.image_transform = image_transform

    def prepare_image(self, img: Union[str, bytes]) -> Tensor:
        image = np.array(Image.open(img))
        image = self.image_transform(image=image)["image"]
        return image  # type: ignore

    def __getitem__(self, item: int) -> Tuple[Tensor, str]:
        img = self.prepare_image(self.df.iloc[item, self.img_col_idx])
        text = self.df.iloc[item, self.text_col_idx]
        return img, text

    def __len__(self) -> int:
        return self.df.shape[0]


class CLIPDataset(ImageDataset):
    def __init__(
        self,
        dataframe: DataFrame,
        image_transform: A.Compose,
        text_transform: ComposeAugmentator,
        image_column: str = "image",
        text_column: str = "text",
    ) -> None:
        super().__init__(
            dataframe=dataframe,
            image_transform=image_transform,
            image_column=image_column,
            text_column=text_column,
        )
        self.text_transform = text_transform

    def __getitem__(self, item: int) -> Tuple[Tensor, str]:  # type: ignore
        img = self.prepare_image(self.df.iloc[item, self.img_col_idx])
        text: str = self.df.iloc[item, self.text_col_idx]
        return img, self.text_transform(text)


class TextDataset(Dataset):
    def __init__(
        self,
        dataframe: DataFrame,
        text_transform: ComposeAugmentator,
        text_column: str = "text",
    ) -> None:
        self.df = dataframe
        assert text_column in list(self.df.columns), "Write right name for text column"
        self.text_col_idx = list(self.df.columns).index(text_column)
        self.text_transform = text_transform

    def __getitem__(self, item: int) -> str:
        text: str = self.df.iloc[item, self.text_col_idx]
        return self.text_transform(text)

    def __len__(self) -> int:
        return self.df.shape[0]
