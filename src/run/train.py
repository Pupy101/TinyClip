"""
Module with training of CLIP
"""

import os

from os.path import join as path_join
from typing import Any, Optional, Union

import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configurator import Configurator
from ..utils import create_label


def train(configuration: Configurator) -> None:
    """
    Function for training clip with the specified configuration

    :param configuration: configuration of training
    :return: None
    """
    parameters = configuration.train_parameters
    accumulation = parameters['accumulation']
    criterion = parameters['criterion']
    device = parameters['device']
    loaders = parameters['loaders']
    model = parameters['model']
    n_epoch = parameters['n_epoch']
    optimizer = parameters['optimizer']
    save_dir = parameters['save_dir']
    scheduler = parameters['scheduler']

    os.makedirs(save_dir, exist_ok=True)

    min_val_loss = float('inf')
    best_epoch = 0

    for i in range(n_epoch):
        train_loss = train_epoch(
            model=model, dataloader=loaders['train'], criterion=criterion,
            optimizer=optimizer, device=device, scheduler=scheduler,
            accumulation=accumulation
        )
        val_loss = eval_epoch(
            model=model, dataloader=loaders['valid'], criterion=criterion,
            device=device
        )
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = i + 1
            torch.save(
                model.state_dict(),
                path_join(
                    config.PATH_TO_WEIGHTS['PATH_TO_SAVE'],
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
        text_features = None
        if 'text_features' in batch:
            text_features = batch['text_features'].to(device)

        logits_img, logits_text, (_, text_embedding) = model(
            image=image, text=text, text_features=text_features
        )

        labels_img = create_label(text_embedding)
        labels_text = labels_img.clone().t()

        loss = criterion(logits_img, labels_img) + criterion(logits_text, labels_text)
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
        text_features = None
        if 'text_features' in batch:
            text_features = batch['text_features'].to(device)

        logits_img, logits_text, (_, text_embedding) = model(
            image=image, text=text, text_features=text_features
        )

        labels_img = create_label(text_embedding)
        labels_text = labels_img.t()

        loss = criterion(logits_img, labels_img) + criterion(logits_text, labels_text)
        
        eval_loss += loss.item()
        count += 1

    return eval_loss / count
