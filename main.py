from src import Configurator, inference_clip, train_clip

from config import Config


if __name__ == '__main__':
    configuration = Configurator(Config)
    if Config.TYPE_USING == 'train':
        print('Training model:')
        train_clip(configuration)
    elif Config.TYPE_USING == 'eval':
        print('Evaluation model:')
        inference_clip(configuration)
    else:
        raise ValueError('Strange type of using CLIP')
