"""
Module with different datasets
"""

from io import BytesIO
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch


from pandas import DataFrame
from PIL import Image
from requests import Session
from torch.utils.data import Dataset

from src.utils.functions import get_image


class TextAndImageCachedText(Dataset):
    """
    Torch dataset for training CLIP with cached text embedding
    """
    def __init__(
            self,
            csv: DataFrame,
            transform: Optional[Callable] = None,
    ) -> None:
        """
        Method for init dataset

        Args:
            csv: dataframe with many columns: first - path to image; second -
                text description; other columns are vector representation
            transform: augmentation for image
        """
        self.csv = csv
        self.transform = transform

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting a pair of image and text

        Args:
            item: index of item

        Returns:
            dict with two keys: image and text
        """
        file_name = self.csv.iloc[item, 0]
        img = self.prepare_image(file_name)
        text_features_numpy = self.csv.iloc[item, 2:].to_numpy('float32')
        text_features = torch.from_numpy(text_features_numpy).float()
        return {'image': img, 'text': torch.tensor(1), 'text_features': text_features}

    def prepare_image(self, img: Union[str, bytes]) -> torch.Tensor:
        image = np.array(Image.open(img))
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]


class TextAndImage(TextAndImageCachedText):
    """
    Torch dataset for training CLIP
    """
    def __init__(
            self,
            csv: DataFrame,
            tokenizer: Callable,
            max_seq_len: int,
            transform: Optional[Callable] = None,
    ) -> None:
        """
        Method for init dataset

        Args:
            csv: dataframe with 2 columns: first - path to image
                and second - text description
            tokenizer: tokenizer for text
            max_seq_len: max length for token sequence
            transform: augmentation for image
        """
        super().__init__(csv=csv, transform=transform)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting a pair of image and text

        Args:
            item: index of item

        Returns:
            dict with two keys: image and text
        """
        file_name = self.csv.iloc[item, 0]
        img = self.prepare_image(file_name)

        text = self.csv.iloc[item, 1]
        tokenized_text = self.prepare_text(text)

        return {'image': img, 'text': tokenized_text}

    def prepare_text(self, text: str):
        tokenized_text = self.tokenizer(
            text, return_tensors="pt"
        )['input_ids'].squeeze(0)[:self.max_seq_len]
        if self.max_seq_len - len(text):
            padding = torch.tensor(
                [0] * (self.max_seq_len - len(text)), dtype=torch.int
            )
            tokenized_text = torch.cat([tokenized_text, padding])
        return tokenized_text

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]


class InferenceImage(TextAndImageCachedText):
    """
    Torch dataset for inference CLIP with image
    """
    def __init__(
            self,
            csv: DataFrame,
            transform: Optional[Callable] = None,
    ) -> None:
        """
        Method for init dataset

        Args:
            csv: dataframe with 1 column - path to image
            transform: augmentation for image
        """
        super().__init__(csv=csv, transform=transform)

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting image and it's index in dataframe

        Args:
            item: index of item

        Returns:
            dict with two keys: image and index
        """
        file_name = self.csv.iloc[item, 0]
        img = self.prepare_image(file_name)
        return {'image': img, 'index': torch.tensor(item).long()}

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]


class TextAndImageURL(TextAndImage):

    def __init__(
            self,
            csv: DataFrame,
            tokenizer: Callable,
            max_seq_len: int,
            session: Session,
            transform: Optional[Callable] = None,
    ):
        """
        Function for learning on dataset with image as url

        Args:
            csv: csv with two columns url and description
            tokenizer: tokenizer
            max_seq_len: max sequence length
            session: requests session
            transform: transform for image
        """
        super().__init__(
            csv=csv, tokenizer=tokenizer, max_seq_len=max_seq_len, transform=transform
        )
        self.session = session

    def __len__(self):
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]

    def __getitem__(self, item):
        """
        Method for getting a pair of image and text

        Args:
            item: index of item

        Returns:
            dict with two keys: image and text
        """

        url = self.csv.iloc[item, 0]
        img = get_image(session=self.session, url=url)
        if img is not None:
            img = self.prepare_image(BytesIO(img))

        text = self.csv.iloc[item, 1]
        tokenized_text = self.prepare_text(text)

        return {'image': img, 'text': tokenized_text}
