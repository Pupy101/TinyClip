import logging

from config import (
    clip_shape,
    data_config,
    image_model_config,
    text_model_config,
    train_config,
)
from src.engine import Configurator, train
from src.models import ConvNeXt, TextPartCLIP, VisionPartCLIP, XLNet

logging.basicConfig(
    format="%(message)s", filename="train.log", filemode="w", level=logging.INFO
)

vision_net = ConvNeXt(config=image_model_config)
text_net = XLNet(config=text_model_config)


configurator = Configurator(
    vision_part=VisionPartCLIP(
        model=vision_net,
        output_model_shape=clip_shape,
        count_classes=data_config.image_classification_count_classes,
    ),
    text_part=TextPartCLIP(
        model=text_net,
        output_model_shape=text_model_config.model_dim,
        count_tokens=text_model_config.vocab_size,
        output_shape=clip_shape,
    ),
    data_config=data_config,
    train_config=train_config,
)

train_parameters = configurator.configurate()

train(parameters=train_parameters)
