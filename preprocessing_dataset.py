import os
import json
import argparse

from typing import Dict
from os.path import join as path_join

import pandas as pd

from sklearn.model_selection import train_test_split

# this scrip is write for dataset from kaggle:
# https://www.kaggle.com/mrviswamitrakaushik/image-captioning-data
# https://www.kaggle.com/ashish2001/original-flickr8k-dataset
# https://www.kaggle.com/adityajn105/flickr30k


def check_file_exist(path):
    return os.path.exists(path)


def create_csv_coco(data: dict, directory: Dict[str, str]) -> pd.DataFrame:
    """
    Function for create csv with 2 columns - path to image and text
        description from coco dataset
    :param data: json data about dataframe
    :param directory: directories with images
    :return: pandas.DataFrame
    """
    path_to_images = []
    text_descriptions = []
    images = data['images']
    print('Preprocess COCO dataset')
    for image in images:
        image_folder = image['filename'].split('_')[1]
        text_description = [_['raw'] for _ in image['sentences']]
        if image_folder == 'train2014':
            path_to_image = path_join(directory['train'], image['filename'])
        else:
            path_to_image = path_join(directory['val'], image['filename'])
        if not check_file_exist(path_to_image):
            continue
        path_to_images.extend([path_to_image] * len(text_description))
        text_descriptions.extend(text_description)
    return pd.DataFrame({'image': path_to_images, 'text': text_descriptions})


def create_csv_flickr(data: dict, directory: str) -> pd.DataFrame:
    """
    Function for create csv with 2 columns - path to image and text
        description from flickr dataset
    :param data: json data about dataframe
    :param directory: directory with images
    :return: pandas.DataFrame
    """
    path_to_images = []
    text_descriptions = []
    images = data['images']
    print('Preprocess Flickr dataset')
    for image in images:
        text_description = [_['raw'] for _ in image['sentences']]
        path_to_image = path_join(directory, image['filename'])
        if not check_file_exist(path_to_image):
            continue
        path_to_images.extend([path_to_image] * len(text_description))
        text_descriptions.extend(text_description)
    return pd.DataFrame({'image': path_to_images, 'text': text_descriptions})


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
    args = parser.parse_args()
    dataset_arguments = {}
    for file in os.listdir(args.dir_jsons):
        dataset_name = file.split('_')[1].split('.')[0]
        file_json = path_join(args.dir_jsons, file)
        if dataset_name == 'coco':
            dataset_arguments['coco'] = {
                'func': create_csv_coco, 'json': file_json,
                'directory': {
                    'train': args.coco_train, 'val': args.coco_valid
                }
            }
        elif dataset_name == 'flickr8k':
            dataset_arguments['flickr8k'] = {
                'func': create_csv_flickr,
                'json': file_json,
                'directory': args.flickr8k
            }
        elif dataset_name == 'flickr30k':
            dataset_arguments['flickr30k'] = {
                'func': create_csv_flickr,
                'json': file_json,
                'directory': args.flickr30k
            }
    df = None
    for key in dataset_arguments:
        func = dataset_arguments[key]['func']
        dirs = dataset_arguments[key]['directory']
        with open(dataset_arguments[key]['json']) as f:
            json_data = json.load(f)
            if df is None:
                df = func(json_data, dirs)
            else:
                df = pd.concat([
                    df,
                    func(json_data, dirs)
                ], axis=0)
    print(f'Overall count of pairs image-text before filter is {df.shape[0]}')
    df = df.drop_duplicates(subset=['text'])
    print(f'Overall count of pairs image-text after filter is {df.shape[0]}')

    grouped = df.groupby(by='image', as_index=False).agg({'text': 'count'})

    # train/valid split
    train_images, valid_images = train_test_split(
        grouped.image, random_state=42, train_size=0.8, stratify=grouped.images
    )
    index_train = df.image.isin(train_images)
    index_valid = df.image.isin(valid_images)
    train, valid = df[index_train], df[index_valid]
    train.to_csv(path_join(args.target_csv, 'train.csv'), index=False)
    valid.to_csv(path_join(args.target_csv, 'valid.csv'), index=False)
