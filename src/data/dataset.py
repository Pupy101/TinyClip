import abc
from typing import List, Tuple

from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset


class BaseImageDataset(Dataset):
    def __init__(self, data: DataFrame) -> None:
        columns = list(data.columns)
        assert "image" in columns, "Not found column 'image' in data"
        self.data = data
        self.img_idx = columns.index("image")

    def get_image(self, index: int) -> Image.Image:
        image_path = self.data.iloc[index, self.img_idx]
        return Image.open(image_path)

    def __len__(self) -> int:
        return self.data.shape[0]

    @abc.abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError


class CLIPDataset(BaseImageDataset):
    def __init__(self, data: DataFrame) -> None:
        super().__init__(data=data)
        columns = list(data.columns)
        assert "text" in columns, "Not found column 'text' in data"
        self.text_idx = columns.index("text")

    def get_text(self, index: int) -> str:
        return self.data.iloc[index, self.text_idx]

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        return self.get_image(index), self.get_text(index)


class EvalCLIPDataset(BaseImageDataset):
    def __init__(self, data: DataFrame, labels: List[str]) -> None:
        super().__init__(data=data)
        columns = list(data.columns)
        assert "label" in columns, "Not found column 'label' in dataframe"
        self.label_idx = columns.index("label")
        self.labels = labels

    def get_label(self, index: int) -> int:
        return self.data.iloc[index, self.label_idx]

    def __getitem__(self, index: int) -> Tuple[Image.Image, List[str], int]:
        return self.get_image(index), self.labels, self.get_label(index)
