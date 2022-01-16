"""
Module with inference of CLIP
"""

import os
import re

import pandas as pd
import torch

from os.path import join as join_path

from tqdm import tqdm

from ..configurator import Configurator


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


def inference(configuration: Configurator):
    parameters = configuration.eval_parameters
    classes = parameters['classes']
    csv = parameters['csv']
    device = parameters['device']
    loaders = parameters['loaders']
    model = parameters['model']
    target_dir = parameters['target_dir']
    tokenizer = parameters['tokenizer']

    os.makedirs(target_dir, exist_ok=True)

    model = model.to(device)

    text_features = torch.cat([
        model.text_model(
            **tokenizer(class_description, return_tensors="pt").to(device)
        )
        for class_description in classes
    ]).to(device)
    index_predict_file = find_max_predict_index(target_dir, 'predict')

    images_names = []
    predicted_classes = []

    for batch in tqdm(loaders['valid'], leave=False):
        image, index = batch['image'].to(device), batch['index'].numpy()
        output_classes, (_, text_features) = model.inference(
            image=image, text=None, text_features=text_features,
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
    prediction.to_csv(
        join_path(target_dir, f'predict_{index_predict_file}.csv'),
        index=False
    )
