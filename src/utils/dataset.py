from typing import Callable, Dict, Optional

import cv2
import torch

from pandas import DataFrame
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

        :param csv: pandas.DataFrame with 2 columns: first - path to image;
        second - text description.
        :param tokenizer: tokenizer for text
        :param max_seq_len: max length for token sequence
        :param transform: augmentation for image
        """
        self.csv = csv
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting a pair of image and text

        :param item: index of item
        :return: dict with two keys: image and text
        """
        file_name = self.csv.iloc[item, 0]
        img = cv2.cvtColor(
            cv2.imread(file_name),
            cv2.COLOR_BGR2RGB
        )
        if self.transform is not None:
            img = self.transform(image=img)['image']

        description = self.csv.iloc[item, 1]
        text = self.tokenizer(
            description,
            return_tensors="pt"
        )['input_ids'].squeeze(0)[:self.max_seq_len]
        padding_count = self.max_seq_len - len(text)
        if padding_count:
            text = torch.cat([
                text,
                torch.tensor([0] * padding_count, dtype=torch.int)
            ])

        return {
            'image': img,
            'text': text,
        }

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        :return: count of pairs
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
    ) -> None:
        """
        Method for init dataset

        :param csv: pandas.DataFrame with many columns: first - path to image;
        second - text description; other columns are vector representation
        of text description
        :param transform: augmentation for image
        """
        self.csv = csv
        self.transform = transform

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting a pair of image and text

        :param item: index of item
        :return: dict with two keys: image and text
        """
        file_name = self.csv.iloc[item, 0]
        img = cv2.cvtColor(
            cv2.imread(file_name),
            cv2.COLOR_BGR2RGB
        )
        if self.transform is not None:
            img = self.transform(image=img)['image']
        text_features = torch.tensor(self.csv.iloc[item, 2:]).float()
        return {
            'image': img,
            'text': [1],
            'text_features': text_features
        }

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        :return: count of pairs
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
    ):
        """
        Method for init dataset

        :param csv: pandas.DataFrame with 1 column - path to image
        :param transform: augmentation for image
        """
        self.csv = csv
        self.transform = transform

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Method for getting image and it's index in pandas.DataFrame

        :param item: index of item
        :return: dict with two keys: image and index
        """
        img = cv2.cvtColor(
            cv2.imread(self.csv.iloc[item, 0]),
            cv2.COLOR_BGR2RGB
        )
        if self.transform is not None:
            img = self.transform(image=img)['image']

        return {
            'image': img,
        }

    def __len__(self) -> int:
        """
        Method for getting count of pairs

        :return: count of pairs
        """
        return self.csv.shape[0]
