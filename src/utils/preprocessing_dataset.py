"""
Script for prepare dataset
"""

import os
import json
import argparse

from typing import Callable, Dict, Union
from os.path import join as path_join

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from config import Config

# this scrip is write for dataset from kaggle:
# https://www.kaggle.com/mrviswamitrakaushik/image-captioning-data
# https://www.kaggle.com/ashish2001/original-flickr8k-dataset
# https://www.kaggle.com/adityajn105/flickr30k


def create_csv(
        data: dict, directory: Union[str, Dict[str, str]]
) -> pd.DataFrame:
    """
    Function for create csv with 2 columns - path to image and text description
    from coco dataset

    :param data: json data about dataframe
    :param directory: directories with images
    :return: pandas.DataFrame
    """
    path_to_images = []
    text_descriptions = []
    images = data['images']
    for image in images:
        text_description = [_['raw'] for _ in image['sentences']]
        if 'train2014' in image['filename']:
            path_to_image = path_join(directory['train'], image['filename'])
        elif 'val2014' in image['filename']:
            path_to_image = path_join(directory['val'], image['filename'])
        else:
            path_to_image = path_join(directory, image['filename'])
        if not os.path.exists(path_to_image):
            continue
        path_to_images.extend([path_to_image] * len(text_description))
        text_descriptions.extend(text_description)
    return pd.DataFrame({'image': path_to_images, 'text': text_descriptions})


def cache_text_embedding(
        dataframe: pd.DataFrame,
        model: nn.Module,
        tokenizer: Callable,
        batch_size: int,
) -> pd.DataFrame:
    """
    Function for preprocessing text data and cache embedding into csv
    :param dataframe:
    :param model:
    :param tokenizer:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    df = dataframe.copy()
    df['input'] = df['text'].apply(lambda x: tokenizer(x, return_tensors='pt'))
    df['size'] = df['input'].apply(lambda x: x['input_ids'].shape[1])
    unique_size = df.size.unique().to_list()
    columns = [f'col_{i+1}' for i in range(768)]
    for size in tqdm(unique_size, leave=False):
        texts = dataframe.text[df.size == size]
        for i in range(len(texts) // batch_size + 1):
            batch_text = texts[i*batch_size: (i+1)*batch_size].to_list()
            encoded_input = tokenizer(
                batch_text, return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                model_output = model(**encoded_input).cpu().numpy()
            for text in batch_text:
                index = dataframe.text == text
                dataframe.loc[index, columns] = np.repeat(
                    model_output, np.sum(index), axis=0
                )
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_jsons', type=str, help='Directory with jsons')
    parser.add_argument(
        '--coco_train', type=str, help='Directory with training images'
    )
    parser.add_argument(
        '--coco_valid', type=str, help='Directory with valid images'
    )
    parser.add_argument(
        '--flickr8k', type=str, help='Directory with images from flickr8k'
    )
    parser.add_argument(
        '--flickr30k', type=str, help='Directory with images from flickr30k'
    )
    parser.add_argument(
        '--target_csv', type=str, help='Directory for train and valid .csv'
    )
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()
    dataset_arguments = {}
    for file in os.listdir(args.dir_jsons):
        dataset_name = file.split('_')[1].split('.')[0]
        file_json = path_join(args.dir_jsons, file)
        if dataset_name == 'coco':
            dataset_arguments['coco'] = {
                'json': file_json,
                'directory': {
                    'train': args.coco_train, 'val': args.coco_valid
                },
            }
        elif dataset_name == 'flickr8k':
            dataset_arguments['flickr8k'] = {
                'json': file_json, 'directory': args.flickr8k,
            }
        elif dataset_name == 'flickr30k':
            dataset_arguments['flickr30k'] = {
                'json': file_json, 'directory': args.flickr30k,
            }
    df = None
    for key in dataset_arguments:
        dirs = dataset_arguments[key]['directory']
        with open(dataset_arguments[key]['json']) as f:
            json_data = json.load(f)
            if df is None:
                df = create_csv(json_data, dirs)
            else:
                df = pd.concat([
                    df, create_csv(json_data, dirs)
                ], axis=0)
    print(f'Overall count of pairs image-text before filter is {df.shape[0]}')
    df = df.drop_duplicates(subset=['text'])
    print(f'Overall count of pairs image-text after filter is {df.shape[0]}')

    grouped = df.groupby(by='image', as_index=False).agg({'text': 'count'})

    one_images = grouped['image'][grouped.text == 1]

    # train/valid split
    train, valid = train_test_split(
        df[~df.image.isin(one_images)], random_state=42,
        train_size=0.78, stratify=df[~df.image.isin(one_images)].image
    )
    train = pd.concat([
        train,
        df[df.image.isin(one_images)]
    ])
    print(f'Train samples is {train.shape[0]}')
    print(f'Eval samples is {valid.shape[0]}')
    if Config.DATASET_WITH_CACHED_TEXT:
        train = cache_text_embedding(
            train, Config.MODEL_TEXT, Config.TOKENIZER,
            Config.LOADER_PARAMS['valid']['batch_size'],
        )
        valid = cache_text_embedding(
            valid, Config.MODEL_TEXT, Config.TOKENIZER
        )
    train.to_csv(path_join(args.target_csv, 'train.csv'), index=False)
    valid.to_csv(path_join(args.target_csv, 'valid.csv'), index=False)
