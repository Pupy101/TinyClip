"""
Script with training of CLIP
"""

import os

from os.path import join as path_join
from typing import Any, Optional, Union

import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configurator import Configurator
from ..utils.misc import create_label


def train(configuration: Configurator) -> None:
    """
    Function for training clip with the specified configuration

    Args:
        configuration: configuration of model, and it's using parameters

    Returns:
        None
    """
    params = configuration.train_parameters
    model = params['model']

    os.makedirs(params['save_dir'], exist_ok=True)

    min_val_loss = float('inf')
    best_epoch = 0

    for i in range(params['n_epoch']):
        train_loss = train_epoch(
            model=model,
            loader=params['loaders']['train'],
            criterion=params['criterion'],
            optimizer=params['optimizer'],
            device=params['device'],
            scheduler=params['scheduler'],
            accumulation=params['accumulation'],
        )
        val_loss = eval_epoch(
            model=model,
            loader=params['loaders']['valid'],
            criterion=params['criterion'],
            device=params['device'],
        )
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = i + 1
            torch.save(
                model.state_dict(),
                path_join(params['save_dir'], f'Model_epoch_{best_epoch}.pth'),
            )
        print(
            f'Epoch {i+1:<3}\t'
            f'Train loss: {train_loss:<10.4f}\tValid loss: {val_loss:<10.4f}'
        )
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss:<10.4f}')


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: Union[str, torch.device],
        scheduler: Optional[Any] = None,
        accumulation: Optional[int] = 1,
) -> Union[int, float]:
    """
    Function for train model on one epoch

    Args:
        model: CLIP
        loader: torch DataLoader
        criterion: criterion for training
        optimizer: optimizer for training
        device: device for training
        scheduler: scheduler or None
        accumulation: count accumulation batches or None

    Returns:
        mean train loss on epoch
    """
    model.train()
    train_loss, count = 0, 1
    acc_image_logits, acc_text_logits = [], []
    acc_img_labels, acc_text_labels = [], []
    optimizer.zero_grad()
    for batch in tqdm(loader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        text_features = batch.get('text_features', None)
        if text_features is not None:
            text_features = text_features.to(device)

        if accumulation > 1:
            *_, (image_embedding, text_embedding) = model(
                image=image, text=text, text_features=text_features,
                only_features=True
            )
            logit_scale = model.vision_part.logit_scale.exp()
            image_logit = logit_scale * image_embedding @ text_embedding.t()
            text_logit = image_logit.t()

            img_label = create_label(text_embedding)
            text_label = img_label.clone().t()

            acc_image_logits.append(image_logit)
            acc_text_logits.append(text_logit)
            acc_img_labels.append(img_label)
            acc_text_labels.append(text_label)

            if count % accumulation and count == len(loader):
                continue

            img_logits = torch.cat(acc_image_logits, dim=0)
            text_logits = torch.cat(acc_text_logits, dim=0)

            img_labels = torch.cat(acc_img_labels, dim=0)
            text_labels = torch.cat(acc_text_labels, dim=0)

        else:
            img_logits, text_logits, (image_embedding, text_embedding) = model(
                image=image, text=text, text_features=text_features
            )
            img_labels = create_label(text_embedding)
            text_labels = img_labels.clone().t()

        loss = criterion(img_logits, img_labels) + criterion(text_logits, text_labels)
        loss.backward()

        acc_image_logits, acc_text_logits = [], []
        acc_img_labels, acc_text_labels = [], []

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
        loader: DataLoader,
        criterion: nn.Module,
        device: Union[str, torch.device]
) -> Union[int, float]:
    """
    Function for evaluation model on one epoch

    Args:
        model: CLIP
        loader: torch DataLoader
        criterion: criterion for training
        device: device for evaluation

    Returns:
        mean evaluation loss on epoch
    """
    model.eval()
    eval_loss, count = 0, 0
    for batch in tqdm(loader, leave=False):
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
