from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from pandas import DataFrame
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from youtokentome import BPE


class ImageDataset(Dataset):
    """Dataset from dataframe with columns path to image and label of image."""

    def __init__(
        self,
        dataframe: DataFrame,
        transform: Callable,
        name_image_column: str = "image",
        name_label_column: str = "label",
    ) -> None:
        assert name_image_column in list(
            dataframe.columns
        ), "Write right name for image column"
        assert name_label_column in list(
            dataframe.columns
        ), "Write right name for label column"
        self.df = dataframe
        self.img_col_idx = list(self.df.columns).index(name_image_column)
        self.label_col_idx = list(self.df.columns).index(name_label_column)
        self.transform = transform

    def prepare_image(self, img: Union[str, bytes]) -> Tensor:
        image = np.array(Image.open(img))
        image = self.transform(image=image)["image"]
        return image  # type: ignore

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        img = self.prepare_image(self.df.iloc[item, self.img_col_idx])
        label = self.df.iloc[item, self.label_col_idx]
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return self.df.shape[0]


class CLIPDataset(ImageDataset):
    """Dataset from dataframe with columns path to image and text description."""

    def __init__(
        self,
        dataframe: DataFrame,
        image_transform: Callable,
        tokenizer: BPE,
        name_image_column: str = "image",
        name_text_column: str = "text",
        cls_token_ids: int = 0,
        cls_token_position: int = 0,
        pad_token_ids: int = 1,
        tokens_max_len: int = 20,
    ) -> None:
        super().__init__(
            dataframe=dataframe,
            transform=image_transform,
            name_image_column=name_image_column,
            name_label_column=name_text_column,
        )
        self.tokenizer = tokenizer
        self.cls_idx = cls_token_ids
        self.cls_pos = cls_token_position
        self.pad_idx = pad_token_ids
        self.max_len = tokens_max_len

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        img = self.prepare_image(self.df.iloc[item, self.img_col_idx])
        text = self.df.iloc[item, self.label_col_idx]
        text_ids: List[int] = self.tokenizer.encode(text.lower())[: self.max_len]
        if len(text_ids) < self.max_len:
            text_ids.extend([self.pad_idx for _ in range(self.max_len - len(text_ids))])
        text_ids.insert(self.cls_pos, self.cls_idx)
        return img, torch.tensor(text_ids, dtype=torch.long)


class MaskedLMDataset(Dataset):
    """Dataset from dataframe with column containing text."""

    def __init__(
        self,
        dataframe: DataFrame,
        tokenizer: BPE,
        transform: Callable[[List[int], int, float], Tuple[Tensor, Tensor, Tensor]],
        name_text_column: str = "text",
        mask_token_idx: int = 0,
        pad_token_ids: int = 1,
        tokens_max_len: int = 20,
        masked_portion: float = 0.25,
    ) -> None:
        self.df = dataframe
        assert name_text_column in list(
            self.df.columns
        ), "Write right name for text column"
        self.text_col_idx = list(self.df.columns).index(name_text_column)
        self.tokenizer = tokenizer
        self.transform = transform
        self.mask_idx = mask_token_idx
        self.pad_idx = pad_token_ids
        self.max_len = tokens_max_len
        self.portion = masked_portion

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor]:
        text = self.df.iloc[item, self.text_col_idx]
        tokenized_text: List[int] = self.tokenizer.encode(text.lower())[: self.max_len]
        length = len(tokenized_text)
        if len(tokenized_text) < self.max_len:
            length = len(tokenized_text)
            tokenized_text.extend([self.pad_idx for _ in range(self.max_len - length)])
        input_ids, perm_msk, msk_msk = self.transform(
            tokenized_text, length, self.portion
        )
        masked_ids = input_ids.clone()
        masked_ids[msk_msk.type(torch.long)] = self.mask_idx
        return masked_ids, perm_msk, input_ids.type(torch.long)

    def __len__(self) -> int:
        return self.df.shape[0]
