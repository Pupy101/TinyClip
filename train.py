from pathlib import Path

from dacite import from_dict
from torch import nn, optim
from transformers import BertTokenizerFast
from utilities.data import load_yaml
from utilities.web import configure_ssl

from clip.data import Configurator
from clip.engine import Engine, train
from clip.models import CLIP, ImagePartCLIP, TextPartCLIP
from clip.types import MultiTaskCriterions, MultiTaskDataLoaders, MultiTaskProportions, TrainConfig

CONFIG_PATH = Path(__file__).parent / "config.yaml"
N_EPOCHS = 10
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.02
DEVICE = "cpu"
COUNT_ACCUMULATING_STEPS = 1


def main():
    config = from_dict(data_class=TrainConfig, data=load_yaml(CONFIG_PATH))
    tokenizer = BertTokenizerFast.from_pretrained(config.data.tokenizer)
    loaders = MultiTaskDataLoaders(
        clip=Configurator.create_clip_dataloaders(
            **config.data.clip.__dict__, max_length=config.data.max_length, tokenizer=tokenizer
        ),
        image=Configurator.create_image_dataloaders(**config.data.image.__dict__),
        text=Configurator.create_text_dataloaders(
            **config.data.text.__dict__, max_length=config.data.max_length, tokenizer=tokenizer
        ),
    )
    clip = CLIP(image_part=ImagePartCLIP(**config.model.image), text_part=TextPartCLIP(**config.model.text))
    optimizer = optim.AdamW(params=clip.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterions = MultiTaskCriterions(
        clip=nn.BCEWithLogitsLoss(),
        image=nn.CrossEntropyLoss(),
        text=nn.CrossEntropyLoss(),
    )
    coefficients = MultiTaskProportions(clip=1.0, image=0.2, text=0.2)
    engine = Engine(
        clip=clip,
        criterions=criterions,
        coefficients=coefficients,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=None,
        count_accumulating_steps=COUNT_ACCUMULATING_STEPS,
    )

    train(engine=engine, loaders=loaders, n_epochs=2, save_dir="/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump")


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    configure_ssl()
    main()
