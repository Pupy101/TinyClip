import logging

from config import clip_model, data_config, train_config
from src.data import ConfiguratorData
from src.engine import train

logging.basicConfig(
    format="%(message)s", filename="train.log", filemode="w", level=logging.INFO
)

loaders = ConfiguratorData(data_config=data_config).configurate()

train(config=train_config, clip=clip_model, loaders=loaders)
