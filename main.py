from src import Configurator, inference_clip, train_clip

from config import Config as clip_config


if __name__ == '__main__':
    configuration = Configurator(clip_config)
    if clip_config.TYPE_USING == 'train':
        print('Training model')
        train_clip(configuration)
    elif clip_config.TYPE_USING == 'eval':
        print('Evaluation model')
        inference_clip(configuration)
