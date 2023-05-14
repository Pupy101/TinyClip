from pathlib import Path

import torch
from accelerate import Accelerator
from dacite import from_dict
from torch import nn, optim
from transformers import Adafactor, BertTokenizerFast
from utilities.data import load_yaml
from utilities.web import configure_ssl

from clip.data import Configurator
from clip.loops import train_clip_model
from clip.models import CLIP, ImagePartCLIP, TextPartCLIP
from clip.types import TrainConfig

CONFIG_PATH = Path(__file__).parent / "config.yaml"
N_EPOCHS = 15
WEIGHT_DECAY = 0.02
IMAGE_WEIGHTS_PATH = "/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump/image.pth"
TEXT_WEIGHTS_PATH = "/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump/text.pth"
SAVE_PATH = "/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump/clip.pth"


def main():
    config = from_dict(data_class=TrainConfig, data=load_yaml(CONFIG_PATH))
    tokenizer = BertTokenizerFast(config.data.tokenizer)
    loaders = Configurator.create_clip_dataloaders(
        **config.data.clip.__dict__, max_length=config.data.max_length, tokenizer=tokenizer
    )
    image_part = ImagePartCLIP(**config.model.image)
    image_part.load_state_dict(torch.load(IMAGE_WEIGHTS_PATH))
    text_part = TextPartCLIP(**config.model.text)
    text_part.load_state_dict(torch.load(TEXT_WEIGHTS_PATH))
    clip = CLIP(image_part=image_part, text_part=text_part)
    optimizer = Adafactor(params=clip.parameters(), weight_decay=WEIGHT_DECAY)

    accelerator = Accelerator()
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    clip, optimizer, train_loader, valid_loader, test_loader, scheduler = accelerator.prepare(
        clip, optimizer, loaders.train, loaders.valid, loaders.test, scheduler
    )
    clip = train_clip_model(
        accelerator=accelerator,
        n_epochs=N_EPOCHS,
        clip=clip,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        scheduler=scheduler,
    )
    clip.to("cpu")
    torch.save(clip.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    configure_ssl()
    main()
