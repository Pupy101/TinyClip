from typing import Union

import hydra
from accelerate import Accelerator
from omegaconf import DictConfig

from clip.data import CLIPConfigurator, ImageConfigurator, TextConfigurator
from clip.loops import train_clip_model, train_image_model, train_text_model
from clip.models import CLIP, ImagePart, TextPart, TextPartMLM
from clip.types import ImageModel, TextModel, TrainMode

Model = Union[CLIP, ImagePart, TextPart, TextPartMLM]


@hydra.main(config_path="configs/", config_name="train.yaml", version_base="1.2")
def train(config: DictConfig):
    accelerator: Accelerator = hydra.utils.instantiate(config.accelerator)
    if config.mode == TrainMode.CLIP.value:
        text_part: TextPart = hydra.utils.instantiate(config.model.text)
        image_part: ImagePart = hydra.utils.instantiate(config.model.image)
        model = CLIP(image_part=image_part, text_part=text_part)
    elif config.mode in TrainMode.IMAGE.value:
        text_part: TextPart = hydra.utils.instantiate(config.model.text)
        model = hydra.utils.instantiate(config.model.image)
    elif config.mode == TrainMode.TEXT.value:
        config.model.text._target_ = "clip.models.TextPartMLM"  # pylint: disable=protected-access
        model: TextPartMLM = hydra.utils.instantiate(config.model.text)
    else:
        raise ValueError(f"Strange train mode: {config.mode}")


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
