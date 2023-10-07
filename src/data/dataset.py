from typing import List, Optional, Tuple

import albumentations as A
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from utilities.data import load_image

from src.utils import RankedLogger

log = RankedLogger(__name__, is_rank_zero_only=True)


class CLIPDataset(Dataset):
    def __init__(self, dataframe: DataFrame, transform: A.Compose) -> None:
        self.df = dataframe
        columns = list(dataframe.columns)
        assert "image" in columns, "Not found column 'image' in dataframe"
        self.img_idx = columns.index("image")
        assert "ru_text" in columns or "en_text" in columns, "Not found column 'ru_text' or 'en_text' in dataframe"
        self.ru_idx = columns.index("ru_text")
        self.en_idx = columns.index("en_text")
        self.transform = transform

    def prepare_image(self, img: str) -> Optional[Tensor]:
        try:
            image = load_image(img)
            image = self.transform(image=image)["image"]
            return image  # type: ignore
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception("Catch exception:")
        return None

    def __getitem__(self, idx: int) -> Tuple[Optional[Tensor], List[str]]:  # type: ignore
        img = self.prepare_image(self.df.iloc[idx, self.img_idx])
        texts: List[str] = []
        ru_text = self.df.iloc[idx, self.ru_idx]
        en_text = self.df.iloc[idx, self.en_idx]
        if isinstance(ru_text, str):
            texts.append(ru_text)
        if isinstance(en_text, str):
            texts.append(en_text)

        return img, texts

    def __len__(self) -> int:
        return self.df.shape[0]
