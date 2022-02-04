from src import Configurator
from src.engine import inference, train

from config import Config


if __name__ == '__main__':
    configuration = Configurator(Config)
    if Config.TYPE_USING == 'train':
        print('Training model:')
        train(configuration)
    elif Config.TYPE_USING == 'eval':
        print('Evaluation model:')
        inference(configuration)
    else:
        raise ValueError('Strange type of using CLIP')
