import os
import re

import cv2
import pandas as pd
import torch

from os.path import join as join_path

from torch import nn
from tqdm import tqdm

from .configurator import Configurator


def find_max_predict_index(dir_to_predict, prefix_to_file):
    """
    Function for search last prediction and return it index
    :param dir_to_predict: directory for predict
    :param prefix_to_file: prefix for file with predictions
    :return: last index of file with predictions
    """
    pattern = f'{prefix_to_file}_([0-9]+)'
    files = os.listdir(dir_to_predict)
    max_file_index = 0
    for file in files:
        name, _ = os.path.splitext(file)
        match = re.match(pattern, name)
        if match:
            max_file_index = max(max_file_index, int(match.group(1)))
    return max_file_index + 1


def inference_clip(configuration: Configurator):
    parameters = configuration.eval_parameters
    classes = parameters['classes']
    config = parameters['config']
    csv = parameters['csv']
    loaders = parameters['loaders']
    model = parameters['model']
    target_dir = parameters['target_dir']
    tokenizer = parameters['tokenizer']
    os.makedirs(target_dir, exist_ok=True)
    index_predict_file = find_max_predict_index(target_dir, 'predict')
    if config.PATH_TO_WEIGHTS['PRETRAINED_WEIGHTS'] is not None:
        model.load_state_dict(
            torch.load(config.PATH_TO_WEIGHTS['PRETRAINED_WEIGHTS'])
        )

    text = tokenizer(
        classes, return_tensors="pt", padding=True
    )['input_ids']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, text = config.MODEL.to(DEVICE), text.to(DEVICE)

    images_names = []
    predicted_classes = []
    text_features = None

    for batch in loaders['valid']:
        image, index = batch['image'].to(DEVICE), batch['index'].numpy()
        output_classes, (_, text_features) = model.inference(
            image=image, text=text, text_features=text_features,
            is_raw_output=True
        )
        images_names.extend(
            csv.iloc[index, 0].to_list()
        )
        predicted_classes.extend(
            output_classes.view(-1).cpu().tolist()
        )
    prediction = pd.DataFrame({
        'file_name': images_names,
        'Class': [classes[i] for i in predicted_classes]
    })
    prediction.to_csv(f'predict_{index_predict_file}.csv')
