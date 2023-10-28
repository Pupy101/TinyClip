from random import choice
from typing import List, Tuple

import albumentations as A
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from utilities.data import load_image

from src.utils import RankedLogger

log = RankedLogger(__name__, is_rank_zero_only=True)


class BaseImageDataset(Dataset):  # pylint: disable=abstract-method
    def __init__(self, dataframe: DataFrame, transform: A.Compose) -> None:
        columns = list(dataframe.columns)
        assert "image" in columns, "Not found column 'image' in dataframe"

        self.df = dataframe
        self.transform = transform
        self.img_idx = columns.index("image")

    def prepare_image(self, img: str) -> Tensor:
        image = load_image(img)
        image = self.transform(image=image)["image"]
        return image  # type: ignore

    def __len__(self) -> int:
        return self.df.shape[0]


class CLIPDataset(BaseImageDataset):
    def __init__(self, dataframe: DataFrame, transform: A.Compose) -> None:
        super().__init__(dataframe=dataframe, transform=transform)

        columns = list(dataframe.columns)
        assert "ru_text" in columns, "Not found column 'ru_text' in dataframe"
        assert "en_text" in columns, "Not found column 'en_text' in dataframe"

        self.ru_idx = columns.index("ru_text")
        self.en_idx = columns.index("en_text")

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, None]:
        img = self.prepare_image(self.df.iloc[idx, self.img_idx])

        texts: List[str] = []
        ru_text = self.df.iloc[idx, self.ru_idx]
        if isinstance(ru_text, str):
            texts.append(ru_text)
        en_text = self.df.iloc[idx, self.en_idx]
        if isinstance(en_text, str):
            texts.append(en_text)

        return img, choice(texts), None


class EvalCLIPDataset(BaseImageDataset):
    def __init__(self, dataframe: DataFrame, transform: A.Compose, labels: List[str]) -> None:
        super().__init__(dataframe=dataframe, transform=transform)

        columns = list(dataframe.columns)
        assert "label" in columns, "Not found column 'label' in dataframe"

        self.labels = labels
        self.label_idx = columns.index("label")

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[str], int]:
        img = self.prepare_image(self.df.iloc[idx, self.img_idx])

        return img, self.labels, self.df.iloc[idx, self.label_idx]
