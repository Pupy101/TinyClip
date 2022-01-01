import os

from os.path import join as path_join
from typing import Any, Optional, Union

import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configurator import Configurator


def train_clip(configuration: Configurator) -> None:
    """
    Function for training clip with the specified configuration
    :param configuration: configuration of training
    :return: None
    """
    parameters = configuration.train_parameters
    config = parameters['config']
    model = parameters['model']
    optimizer = parameters['optimizer']
    scheduler = parameters['scheduler']
    criterion = parameters['criterion']
    loaders = parameters['loaders']

    os.makedirs(config.PATH_TO_WEIGHTS['PATH_TO_SAVE'], exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    min_val_loss = float('inf')
    best_epoch = 0
    accumulation = config.ACCUMULATION_STEPS if config.ACCUMULATE else 1

    for i in range(config.NUM_EPOCH):
        train_loss = train_epoch(
            model=model, dataloader=loaders['train'], criterion=criterion,
            optimizer=optimizer, device=DEVICE, scheduler=scheduler,
            accumulation=accumulation
        )
        val_loss = eval_epoch(
            model=model, dataloader=loaders['valid'], criterion=criterion,
            device=DEVICE
        )
        if val_loss < min_val_loss and val_loss < train_loss:
            min_val_loss = val_loss
            best_epoch = i + 1
            torch.save(
                model.state_dict(),
                path_join(
                    config.PATH_TO_SAVE_MODEL_WEIGHTS,
                    f'Model_epoch_{best_epoch}.pth'
                )
            )
        print(f'Epoch {i+1}\tTrain loss: {train_loss:<10.4f}\tValid image loss: {val_loss:<10.4f}')
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss}')


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: Union[str, torch.device],
        scheduler: Optional[Any] = None,
        accumulation: Optional[int] = None
):
    """
    Function for train model on one epoch
    :param model: CLIP
    :param dataloader: torch DataLoader
    :param criterion: criterion for training
    :param optimizer: optimizer for training
    :param device: device for training
    :param scheduler: scheduler or None
    :param accumulation: count accumulation steps or None
    :return: mean train loss on epoch
    """
    model.train()
    train_loss = 0
    count = 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size_image = image.size(0)

        labels_image = torch.tensor(
            [_ for _ in range(batch_size_image)]
        ).to(device)
        labels_text = labels_image.copy()

        logits_image, logits_text = model(image, text)
        loss = (
            criterion(logits_image, labels_image)
            + criterion(logits_text, labels_text)
        )
        loss.backward()
        # accumulating
        if not (count + 1) % accumulation or count + 1 == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        train_loss += loss.item()
        count += 1

    return train_loss / count


@torch.no_grad()
def eval_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: Union[str, torch.device]
):
    """
    Function for evaluation model on one epoch
    :param model: CLIP
    :param dataloader: torch DataLoader
    :param criterion: criterion for training
    :param device: device for evaluation
    :return: mean evaluation loss on epoch
    """
    model.eval()
    eval_loss = 0
    count = 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size_image = image.size(0)

        labels_image = torch.tensor(
            [_ for _ in range(batch_size_image)]
        ).to(device)
        labels_text = labels_image.copy()

        logits_image, logits_text = model(image, text)

        loss = (
                criterion(logits_image, labels_image)
                + criterion(logits_text, labels_text)
        )
        
        eval_loss += loss.item()
        count += 1

    return eval_loss / count
