"""
Script with inference of CLIP
"""

import os

import pandas as pd
import torch

from os.path import join as join_path

from tqdm import tqdm

from ..configurator import Configurator
from ..utils.functions import find_max_predict_index


def inference(configuration: Configurator) -> None:
    """
    Function for inference CLIP with the specified configuration

    Args:
        configuration: configuration of model, and it's using parameters

    Returns:
       None
    """
    params = configuration.eval_parameters
    device = params['device']
    model = params['model'].to(device)
    tokenizer = params['tokenizer']

    os.makedirs(params['target_dir'], exist_ok=True)
    index_predict_file = find_max_predict_index(
        params['target_dir'], 'prediction'
    )

    text_features = torch.cat([
        model.text_model(
            **tokenizer(class_description, return_tensors="pt").to(device)
        )
        for class_description in params['classes']
    ])

    images_names = []
    predicted_classes = []

    for batch in tqdm(params['loaders']['valid'], leave=False):
        image, index = batch['image'].to(device), batch['index'].numpy()
        output_classes, _ = model.inference(
            image=image, text=None, text_features=text_features
        )
        images_names.extend(params['csv'].iloc[index, 0].to_list())
        predicted_classes.extend(output_classes.view(-1).cpu().tolist())
    prediction = pd.DataFrame({
        'file_name': images_names,
        'Class': [params['classes'][i] for i in predicted_classes]
    })
    prediction.to_csv(
        join_path(params['target_dir'], f'predict_{index_predict_file}.csv'),
        index=False
    )
