import argparse

from src import Configurator
from src.engine import inference, train

from config import Config as clip_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to file with CLIP config')
    args = parser.parse_args()

    configuration = Configurator(args.config)
    if clip_config.TYPE_USING.value == 'train':
        print('Training model:')
        train(configuration)
    elif clip_config.TYPE_USING.value == 'evaluation':
        print('Evaluation model:')
        inference(configuration)
    else:
        raise ValueError('Strange type of using CLIP')
