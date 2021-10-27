import os
import json

from typing import Callable
from os.path import join as join_path

import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class TextAndImageFromCSV(Dataset):

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
            cv2.imread(self.csv['image_name'].iat[item]),
            cv2.COLOR_BGR2RGB
        )
        if self.transform is not None:
            img = self.transform(image=img)['image']

        texts = self.csv['comment'].iat[item]
        marks = 5 - np.array(self.csv['comment_number'].iat[item])
        probability = marks / np.sum(marks)
        text = np.random.choice(texts, p=probability)
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

    def __len__(self):
        return self.csv.shape[0]


def create_datasets_from_csv(
        path_to_csv: str,
        dir_image: str,
        tokenizer: Callable,
        max_size_seq_len: int,
        transform: Callable,
        df: pd.DataFrame = None
):
    if df is None:
        df = pd.read_csv(path_to_csv, delimiter='|')
    df.columns = [x.strip() for x in df.columns]
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if df is None:
        index = df['comment_number'].str.isdigit()
        df = df[index]
    df['image_name'] = df['image_name'].apply(lambda x: join_path(dir_image, x))
    df['comment_number'] = pd.to_numeric(df['comment_number'])
    df = df.groupby(by='image_name', as_index=False).agg(
        {'comment_number': list, 'comment': list}
    )
    df.sample(frac=1).reset_index(drop=True)
    num_example = df.shape[0]
    train_df, valid_df = df.iloc[:round(0.8*num_example), :], df.iloc[round(0.8*num_example):, :]
    return (
        TextAndImageFromCSV(
            csv=train_df,
            tokenizer=tokenizer,
            max_size_seq_len=max_size_seq_len,
            transform=transform['train']
        ),
        TextAndImageFromCSV(
            csv=valid_df,
            tokenizer=tokenizer,
            max_size_seq_len=max_size_seq_len,
            transform=transform['valid']
        )
    )


def create_datasets_from_json(
    jsons: dict,
    dir_image: str,
    tokenizer: Callable,
    max_size_seq_len: int,
    transform: Callable
):
    df = {'image_name': [], 'comment': [], 'comment_number': []}
    df_file_and_sentence = set()
    for json_name in jsons:
        with open(json_name) as f:
            json_description = json.load(f)
        for image in json_description['images']:
            filename = image['filename']
            for description in image['sentences']:
                sentence = description['raw']
                if not (filename, sentence) in df_file_and_sentence and os.path.exists(join_path(dir_image, filename)):
                    df_file_and_sentence.add((filename, sentence))
                    df['image_name'].append(filename)
                    df['comment'].append(sentence)
                    if description['sentid'] > 10:
                        df['comment_number'].append(1)
                    else:
                        df['comment_number'].append(description['sentid'])
    return create_datasets_from_csv(
        '',
        dir_image=dir_image,
        tokenizer=tokenizer,
        max_size_seq_len=max_size_seq_len,
        transform=transform,
        df=pd.DataFrame(df)
    )