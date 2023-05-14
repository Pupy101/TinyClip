from pathlib import Path

import torch
from accelerate import Accelerator
from dacite import from_dict
from torch import nn, optim
from transformers import Adafactor, BertTokenizerFast
from utilities.data import load_yaml
from utilities.web import configure_ssl

from clip.data import Configurator
from clip.loops import train_text_model
from clip.models import TextPartCLIP
from clip.types import TrainConfig

CONFIG_PATH = Path(__file__).parent / "config.yaml"
N_EPOCHS = 10
WEIGHT_DECAY = 0.02
DEVICE = "cpu"
SAVE_PATH = "/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump/text.pth"


def main():
    config = from_dict(data_class=TrainConfig, data=load_yaml(CONFIG_PATH))
    tokenizer = BertTokenizerFast(config.data.tokenizer)
    loaders = Configurator.create_text_dataloaders(
        **config.data.text.__dict__,
        max_length=config.data.max_length,
        tokenizer=tokenizer,
    )
    model = TextPartCLIP(**config.model.text)
    optimizer = Adafactor(params=model.parameters(), weight_decay=WEIGHT_DECAY)
    accelerator = Accelerator()
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model, optimizer, train_loader, valid_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, loaders.train, loaders.valid, loaders.test, scheduler
    )

    model = train_text_model(
        accelerator=accelerator,
        n_epochs=N_EPOCHS,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        scheduler=scheduler,
    )

    model.to("cpu")
    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    configure_ssl()
    main()
