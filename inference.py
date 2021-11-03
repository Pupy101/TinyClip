import os
import re

import cv2
import torch

from os.path import join as join_path

from torch import nn

from utils.augmentations import valid_transform


def find_max_predict_index(dir_to_predict, prefix_to_file):
    files = os.listdir(dir_to_predict)
    max_file_index = 0
    for file in files:
        name, ext = os.path.splitext(file)
        match = re.match(r'{}_([0-9]+)'.format(prefix_to_file), name)
        if match:
            index = int(match.group(1))
            if max_file_index < index:
                max_file_index = index
    return max_file_index + 1


def inference(config):
    inference_params = config.INFERENCE_PARAMS
    os.makedirs(inference_params['TARGET_DIR'], exist_ok=True)
    classes = inference_params['TOKENIZER'](inference_params['CLASSES'], return_tensors="pt")['input_ids']
    index_predict_file = find_max_predict_index(inference_params['TARGET_DIR'], 'predict')
    config.MODEL.load_state_dict(torch.load(config.PATH_TO_WEIGHTS))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes = config.MODEL.to(DEVICE), classes.to(DEVICE)
    with open(join_path(inference_params['TARGET_DIR'], f'predict_{index_predict_file}.txt')) as f, torch.no_grad():
        for file in os.listdir(inference_params['IMAGES_DIR']):
            path_to_image = join_path(inference_params['IMAGES_DIR'], file)
            image = cv2.cvtColor(
                cv2.imread(path_to_image),
                cv2.COLOR_BGR2RGB
            )
            input_image = valid_transform(image=image)['image'].unsqueeze(0).to(DEVICE)
            output = model((input_image, classes))
            print(output.shape)
