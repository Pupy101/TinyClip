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

    Args:
        configuration: configuration of model, and it's using parameters

    Returns:
        None
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
                path_join(save_dir, f'Model_epoch_{best_epoch}.pth')
            )
        print(
            f'Epoch {i+1:<3}\t'
            f'Train loss: {train_loss:<10.4f}\tValid loss: {val_loss:<10.4f}'
        )
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss:<10.4f}')


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: Union[str, torch.device],
        scheduler: Optional[Any] = None,
        accumulation: Optional[int] = None
) -> Union[int, float]:
    """
    Function for train model on one epoch

    Args:
        model: CLIP
        dataloader: torch DataLoader
        criterion: criterion for training
        optimizer: optimizer for training
        device: device for training
        scheduler: scheduler or None
        accumulation: count accumulation steps or None

    Returns:
        mean train loss on epoch
    """
    model.train()
    train_loss, count = 0, 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        text_features = batch.get('text_features', None)
        if text_features is not None:
            text_features = text_features.to(device)

        img_logits, text_logits, (_, text_embedding) = model(
            image=image, text=text, text_features=text_features
        )

        img_labels = create_label(text_embedding)
        text_labels = img_labels.clone().t()

        loss = criterion(img_logits, img_labels) + criterion(text_logits, text_labels)
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
) -> Union[int, float]:
    """
    Function for evaluation model on one epoch

    Args:
        model: CLIP
        dataloader: torch DataLoader
        criterion: criterion for training
        device: device for evaluation

    Returns:
        mean evaluation loss on epoch
    """
    model.eval()
    eval_loss, count = 0, 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        text_features = batch.get('text_features', None)
        if text_features is not None:
            text_features = text_features.to(device)

        img_logits, text_logits, (_, text_embedding) = model(
            image=image, text=text, text_features=text_features
        )

        img_labels = create_label(text_embedding)
        text_labels = img_labels.t()

        loss = criterion(img_logits, img_labels) + criterion(text_logits, text_labels)
        
        eval_loss += loss.item()
        count += 1

    return eval_loss / count
