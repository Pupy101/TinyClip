import abc
from typing import Optional, Tuple

import albumentations as A
import numpy as np
from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset


class BaseImageDataset(Dataset):
    def __init__(self, data: DataFrame, transform: Optional[A.Compose] = None) -> None:
        columns = list(data.columns)
        assert "image" in columns, "Not found column 'image' in data"
        self.data = data
        self.img_idx = columns.index("image")
        self.transform = transform

    def get_image(self, index: int) -> np.ndarray:
        image_path = self.data.iloc[index, self.img_idx]
        image = np.array(Image.open(image_path))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self) -> int:
        return self.data.shape[0]

    @abc.abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError


class CLIPDataset(BaseImageDataset):
    def __init__(self, data: DataFrame, transform: Optional[A.Compose] = None) -> None:
        super().__init__(data=data, transform=transform)
        columns = list(data.columns)
        assert "text" in columns, "Not found column 'text' in data"
        self.text_idx = columns.index("text")

    def get_text(self, index: int) -> str:
        return self.data.iloc[index, self.text_idx]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        return self.get_image(index), self.get_text(index)
