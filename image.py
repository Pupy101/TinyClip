from pathlib import Path

import torch
from accelerate import Accelerator
from dacite import from_dict
from torch import Tensor, nn, optim
from torch.nn import functional as F
from transformers import Adafactor, BertTokenizerFast
from utilities.data import load_yaml

from clip.data import Configurator
from clip.loops import train_image_model
from clip.models import ImagePartCLIP, TextPartCLIP
from clip.types import TrainConfig

CONFIG_PATH = Path(__file__).parent / "config.yaml"
N_EPOCHS = 10
WEIGHT_DECAY = 0.02
DEVICE = "cpu"
TEXT_WEIGHTS_PATH = "/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump/text.pth"
SAVE_PATH = "/Users/19891176/Desktop/MyProjects/CLIP/.exp/dump/image.pth"


class MSEWithKLDivLoss(nn.Module):
    def __init__(self, mse_coeff: float, kl_coeff: float) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_coeff = mse_coeff
        self.kl_coeff = kl_coeff

    def forward(self, output: Tensor, labels: Tensor) -> Tensor:
        return self.mse_coeff * self.mse(output, labels) + self.kl_coeff * self.kl_div(
            F.log_softmax(output, dim=1), F.log_softmax(labels, dim=1)
        )


def main():
    config = from_dict(data_class=TrainConfig, data=load_yaml(CONFIG_PATH))
    tokenizer = BertTokenizerFast(config.data.tokenizer)
    loaders = Configurator.create_image_dataloaders(
        **config.data.image.__dict__,
        max_length=config.data.max_length,
        tokenizer=tokenizer,
    )
    image_model = ImagePartCLIP(**config.model.image)
    text_model = TextPartCLIP(**config.model.text)
    text_model.load_state_dict(torch.load(TEXT_WEIGHTS_PATH))
    optimizer = Adafactor(params=image_model.parameters(), weight_decay=WEIGHT_DECAY)
    accelerator = Accelerator()
    criterion = MSEWithKLDivLoss(mse_coeff=0.4, kl_coeff=0.6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    image_model, text_model, optimizer, train_loader, valid_loader, test_loader, scheduler = accelerator.prepare(
        image_model, text_model, optimizer, loaders.train, loaders.valid, loaders.test, scheduler
    )
    image_model = train_image_model(
        accelerator=accelerator,
        n_epochs=N_EPOCHS,
        image_model=image_model,
        text_model=text_model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        scheduler=scheduler,
    )
    image_model.to("cpu")
    torch.save(image_model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    main()
