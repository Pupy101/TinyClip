import os
import re

import cv2
import torch

from os.path import join as join_path

from torch import nn
from tqdm import tqdm

from utils.augmentations import valid


def find_max_predict_index(dir_to_predict, prefix_to_file):
    pattern = f'{prefix_to_file}_([0-9]+)'
    files = os.listdir(dir_to_predict)
    max_file_index = 0
    for file in files:
        name, _ = os.path.splitext(file)
        match = re.match(pattern, name)
        if match:
            max_file_index = max(max_file_index, int(match.group(1)))
    return max_file_index + 1


def inference_clip(config):
    inference_params = config.INFERENCE_PARAMS
    os.makedirs(inference_params['TARGET_DIR'], exist_ok=True)
    classes = inference_params['TOKENIZER'](
        inference_params['CLASSES'], return_tensors="pt"
    )['input_ids']
    index_predict_file = find_max_predict_index(
        inference_params['TARGET_DIR'], 'predict'
    )
    if config.PATH_TO_WEIGHTS is not None:
        config.MODEL.load_state_dict(torch.load(config.PATH_TO_WEIGHTS))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes = config.MODEL.to(DEVICE), classes.to(DEVICE)
    input_images = []

    for file in tqdm(os.listdir(inference_params['IMAGES_DIR']), leave=False):
        path_to_image = join_path(inference_params['IMAGES_DIR'], file)
        image = cv2.cvtColor(
            cv2.imread(path_to_image),
            cv2.COLOR_BGR2RGB
        )
        input_images.append(
            valid(image=image)['image'].unsqueeze(0).to(DEVICE)
        )
        if len(input_images) >= config.LOADER_PARAMS['valid']['batch_size']:
            output = model.inference((
                torch.cat(input_images, dim=0),
                classes
                ))
            input_images = []
            index = output.view(-1).tolist()[0]
    with open(join_path(inference_params['TARGET_DIR'], f'predict_{index_predict_file}.txt'), 'w') as f:    
        f.write('{} {}\n'.format(inference_params['CLASSES'][index], file))
