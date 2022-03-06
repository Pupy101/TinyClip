"""
Script with training of CLIP
"""

import os

from os.path import join as path_join
from typing import Any, Optional, Tuple, Union

import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configurator import Configurator
from ..classes.engine import OneEpochResults
from ..model.clip import CLIP
from ..utils.functions import compute_f1_batch


def train(configuration: Configurator) -> None:
    """
    Function for training clip with the specified configuration

    Args:
        configuration: configuration of model, and it's using parameters

    Returns:
        None
    """
    params = configuration.train_parameters
    model = params.model

    os.makedirs(params.save_dir, exist_ok=True)

    min_val_loss = float('inf')
    best_epoch = 0

    for i in range(1, params.n_epoch + 1):
        # TODO add f1 score
        train_result = train_epoch(
            model=model,
            loader=params.loaders['train'],
            criterion=params.criterion,
            optimizer=params.optimizer,
            device=params.device,
            scheduler=params.scheduler,
            accumulation=params.accumulation,
        )
        eval_result = eval_epoch(
            model=model,
            loader=params.loaders['valid'],
            criterion=params.criterion,
            device=params.device,
        )
        if eval_result.mean_loss < min_val_loss and train_result.mean_loss > eval_result.mean_loss:
            min_val_loss = eval_result.mean_loss
            best_epoch = i
            torch.save(
                model.state_dict(),
                path_join(params['save_dir'], f'Model_epoch_{best_epoch}.pth'),
            )
        print(
            f'Epoch {i:<3}\tTrain loss:{train_result.mean_loss:<7.4f} Precision:{train_result.precision:<7.4f} '
            f'Recall:{train_result.recall:<7.4f} F1:{train_result.f1:<7.4f}'
            f'\tValid loss:{eval_result.mean_loss:<7.4f} Precision:{eval_result.precision:<7.4f} '
            f'Recall:{eval_result.recall:<7.4f} F1:{eval_result.f1:<7.4f}'
        )
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss:<10.4f}')


def train_epoch(
        model: CLIP,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: Union[str, torch.device],
        scheduler: Optional[Any] = None,
        accumulation: Optional[int] = 1,
) -> OneEpochResults:
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
        mean train loss on epoch, recall, precision, f1
    """
    model.train()
    train_loss = 0
    true_positive, false_positive, false_negative = 0, 0, 0
    accum_image_emb, accum_text_emb = [], []

    optimizer.zero_grad()
    from ..model import CLIP
    model: CLIP

    for i, batch in tqdm(enumerate(loader, 1), leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)

        text_features = batch.get('text_features', None)
        if text_features is not None:
            text_features = text_features.to(device)

        labels = torch.diag(torch.ones(image.size(0))).to(device)

        if accumulation > 1:
            output = model(
                image=image, text=text, text_features=text_features, only_features=True
            )
            accum_image_emb.append(output.embeddings.image)
            accum_text_emb.append(output.embeddings.text)

            if i % accumulation and i != len(loader):
                continue

            overall_image_emb = torch.cat(accum_image_emb, dim=0)
            overall_text_emb = torch.cat(accum_text_emb, dim=0)

            image_logits, text_logits = model.cv_model.compute_logit(
                image_embedding=overall_image_emb,
                text_embedding=overall_text_emb,
                logit_scale=model.vision_part.logit_scale,
            )

        else:
            output = model(image=image, text=text, text_features=text_features)
            image_logits, text_logits = output.image_logit, output.text_logit

        loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        accum_image_emb, accum_text_emb = [], []
        train_loss += loss.item()
        tp, fp, fn = compute_f1_batch(image_logits.detach(), labels)
        true_positive += tp
        false_positive += fp
        false_negative += fn

    recall = true_positive / (true_positive + false_positive)
    precision = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return OneEpochResults(mean_loss=train_loss / i, recall=recall, precision=precision, f1=f1)


@torch.no_grad()
def eval_epoch(
        model: CLIP,
        loader: DataLoader,
        criterion: nn.Module,
        device: Union[str, torch.device]
) -> OneEpochResults:
    """
    Function for evaluation model on one epoch

    Args:
        model: CLIP
        loader: torch DataLoader
        criterion: criterion for training
        device: device for evaluation

    Returns:
        mean evaluation loss on epoch, recall, precision, f1
    """
    model.eval()
    eval_loss = 0
    true_positive, false_positive, false_negative = 0, 0, 0
    for i, batch in tqdm(enumerate(loader, 1), leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        text_features = batch.get('text_features', None)
        if text_features is not None:
            text_features = text_features.to(device)

        output = model(image=image, text=text, text_features=text_features)

        labels = torch.diag(torch.ones(image.size(0))).to(device)

        loss = criterion(output.image_logit, labels) + criterion(output.text_logit, labels)

        tp, fp, fn = compute_f1_batch(output.image_logit.detach(), labels)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        
        eval_loss += loss.item()

    recall = true_positive / (true_positive + false_positive)
    precision = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return OneEpochResults(mean_loss=eval_loss / i, recall=recall, precision=precision, f1=f1)
