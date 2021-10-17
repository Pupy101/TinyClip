import re

from typing import Callable
from os.path import join as join_path

import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset


class TextAndImage(Dataset):

    def __init__(
            self,
            csv: pd.DataFrame,
            tokenizer: Callable,
            max_size_seq_len: int,
            transform: Callable = None
    ):
        self.csv = csv
        self.tokenizer = tokenizer
        self.max_size_seq_len = max_size_seq_len
        self.transform = transform

    def __getitem__(self, item):
        img = cv2.cvtColor(
            cv2.imread(self.csv['image'][item]),
            cv2.COLOR_BGR2RGB
        )
        if self.transform is not None:
            img = self.transform(image=img)['image']

        text = self.csv['text'][item]
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt"
        )['input_ids'].squeeze(0)[:self.max_size_seq_len]
        padding_count = self.max_size_seq_len - len(tokenized_text)
        if padding_count:
            tokenized_text = torch.cat(
                [
                    tokenized_text,
                    torch.tensor([0] * padding_count, dtype=torch.int64)
                ]
            )
        return {
            'image': img,
            'text': tokenized_text
        }


def create_dataset(
        image_dir: str,
        dir_to_train_file: str,
        dir_to_valid_file: str,
        dir_to_caption_file: str,
        tokenizer: Callable,
        max_size_seq_len: int,
        transform: Callable
):
    train_images = set()
    with open(dir_to_train_file) as f:
        for line in f:
            train_images.add(line.strip())
    test_images = set()
    with open(dir_to_valid_file) as f:
        for line in f:
            test_images.add(line.strip())
    train_df = {
        'text': [],
        'image': []
    }
    test_df = {
        'text': [],
        'image': []
    }
    with open(dir_to_caption_file) as f:
        for line in f:
            img, description = re.findall(r'^([\w]+.jpg)#[0-9]+\t(.+.)$', line.strip())
            img_path = join_path(image_dir, img)
            if img_path in train_images:
                if img_path in train_df:
                    train_df[img_path].append(description)
                else:
                    train_df[img_path] = [description]
            else:
                if img_path in test_df:
                    test_df[img_path].append(description)
                else:
                    test_df[img_path] = [description]
    train_csv = pd.DataFrame(train_df)
    test_csv = pd.DataFrame(test_df)
    return (
        TextAndImage(
            csv=train_csv,
            tokenizer=tokenizer,
            max_size_seq_len=max_size_seq_len,
            transform=transform['train']
        ),
        TextAndImage(
            csv=test_csv,
            tokenizer=tokenizer,
            max_size_seq_len=max_size_seq_len,
            transform=transform['test']
        )
    )
