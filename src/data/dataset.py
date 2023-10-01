from typing import List, Optional, Tuple, Union

import albumentations as A
import numpy as np
from pandas import DataFrame
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, is_rank_zero_only=True)


class CLIPDataset(Dataset):
    def __init__(self, dataframe: DataFrame, transform: A.Compose) -> None:
        self.df = dataframe
        columns = list(dataframe.columns)
        assert "image" in columns, "Not found column 'image' in dataframe"
        self.img_idx = columns.index("image")
        assert "ru_text" in columns or "en_text" in columns, "Not found column 'ru_text' or 'en_text' in dataframe"
        self.ru_idx = columns.index("ru_text") if "ru_text" in columns else None
        self.en_idx = columns.index("en_text") if "en_text" in columns else None
        self.transform = transform

    def prepare_image(self, img: Union[str, bytes]) -> Optional[Tensor]:
        try:
            image = np.array(Image.open(img))
            image = self.transform(image=image)["image"]
            return image  # type: ignore
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception("Catch exception:")
        return None

    def __getitem__(self, idx: int) -> Tuple[Optional[Tensor], List[str]]:  # type: ignore
        img = self.prepare_image(self.df.iloc[idx, self.img_idx])
        texts: List[str] = []
        if self.ru_idx is not None:
            texts.append(self.df.iloc[idx, self.ru_idx])
        if self.en_idx is not None:
            texts.append(self.df.iloc[idx, self.en_idx])

        return img, texts

    def __len__(self) -> int:
        return self.df.shape[0]
