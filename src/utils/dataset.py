"""
Module with different datasets
"""

from typing import Callable, Dict, Optional

import numpy as np
import torch


from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset


class TextAndImageFromCSV(Dataset):
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
        self.csv = csv
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
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
        img = np.array(Image.open(file_name))
        if self.transform is not None:
            img = self.transform(image=img)['image']

        text = self.csv.iloc[item, 1]
        text = self.tokenizer(
            text, return_tensors="pt"
        )['input_ids'].squeeze(0)[:self.max_seq_len]
        padding_count = self.max_seq_len - len(text)
        if padding_count:
            text = torch.cat([
                text, torch.tensor([0] * padding_count, dtype=torch.int)
            ])

        return {
            'image': img,
            'text': text,
        }

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]


class TextAndImageCachedTextFromCSV(Dataset):
    """
    Torch dataset for training CLIP with cached text embedding
    """
    def __init__(
            self,
            csv: DataFrame,
            transform: Optional[Callable] = None,
            **kwargs
    ) -> None:
        """
        Method for init dataset

        Args:
            csv: dataframe with many columns: first - path to image; second -
                text description; other columns are vector representation
            transform: augmentation for image
            **kwargs: kwargs
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
        img = np.array(Image.open(file_name))
        if self.transform is not None:
            img = self.transform(image=img)['image']
        text_features_numpy = self.csv.iloc[item, 2:].to_numpy('float32')
        text_features = torch.from_numpy(text_features_numpy).float()
        return {
            'image': img,
            'text': torch.tensor(1),
            'text_features': text_features,
        }

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]


class ImageFromCSV(Dataset):
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
        self.csv = csv
        self.transform = transform

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting image and it's index in dataframe

        Args:
            item: index of item

        Returns:
            dict with two keys: image and index
        """
        file_name = self.csv.iloc[item, 0]
        img = np.array(Image.open(file_name))
        if self.transform is not None:
            img = self.transform(image=img)['image']

        return {
            'image': img,
            'index': torch.tensor(item).long(),
        }

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        Returns:
            count of pairs
        """
        return self.csv.shape[0]
