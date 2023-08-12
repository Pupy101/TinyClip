from contextlib import nullcontext
from typing import Iterable, Optional, TypeVar

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding

from clip.models import CLIP, ImagePart, TextPart, TextPartMLM
from clip.types import Scheduler

Model = TypeVar("Model", ImagePart, TextPart, TextPartMLM)
Item = TypeVar("Item")


def get_iterable(it: Iterable[Item], total: int, is_main: bool, leave: Optional[bool] = None) -> Iterable[Item]:
    if is_main:
        return tqdm(it, total=total, leave=leave)
    return it


def train_text_model(
    accelerator: Accelerator,
    n_epochs: int,
    model: Model,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    scheduler: Optional[Scheduler] = None,
) -> Model:
    for i in get_iterable(range(1, n_epochs + 1), total=n_epochs, is_main=accelerator.is_main_process):
        train_loss = run_text_model(
            accelerator=accelerator,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        valid_loss = run_text_model(accelerator=accelerator, model=model, loader=valid_loader, criterion=criterion)
        if accelerator.is_main_process:
            print(f"Epoch: {i:<3}")
            print(f"Train loss: {train_loss:.2f}")
            print(f"Validation loss: {valid_loss:.2f}")
    test_loss = run_text_model(accelerator=accelerator, model=model, loader=test_loader, criterion=criterion)
    if accelerator.is_main_process:
        print(f"Test loss: {test_loss:.2f}")
    return model


def run_text_model(
    accelerator: Accelerator,
    model: Model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Scheduler] = None,
) -> float:
    context = nullcontext if optimizer else torch.no_grad
    batch: BatchEncoding
    overall_loss = 0
    count_items = 1
    for count_items, batch in get_iterable(
        enumerate(loader, 1),
        total=len(loader),
        is_main=accelerator.is_main_process,
        leave=False,
    ):
        if optimizer:
            optimizer.zero_grad()
        targets = batch.pop("labels")
        with context():
            outputs = model(**batch, masked_lm=True)
        loss = criterion(outputs, targets)
        overall_loss += loss.item()
        if optimizer:
            accelerator.backward(loss)
            optimizer.step()
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(overall_loss / count_items)
    return overall_loss / count_items


def train_image_model(
    accelerator: Accelerator,
    n_epochs: int,
    model: Model,
    text_model: Model,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    scheduler: Optional[Scheduler] = None,
) -> Model:
    for i in get_iterable(range(1, n_epochs + 1), total=n_epochs, is_main=accelerator.is_main_process):
        train_loss = run_image_model(
            accelerator=accelerator,
            model=model,
            text_model=text_model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        valid_loss = run_image_model(
            accelerator=accelerator,
            model=model,
            text_model=text_model,
            loader=valid_loader,
            criterion=criterion,
        )
        if accelerator.is_main_process:
            print(f"Epoch: {i:<3}")
            print(f"Train loss: {train_loss:.2f}")
            print(f"Validation loss: {valid_loss:.2f}")
    test_loss = run_image_model(
        accelerator=accelerator,
        model=model,
        text_model=text_model,
        loader=test_loader,
        criterion=criterion,
    )
    if accelerator.is_main_process:
        print(f"Test loss: {test_loss:.2f}")
    return image_model


def run_image_model(
    accelerator: Accelerator,
    model: Model,
    text_model: Model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Scheduler] = None,
) -> float:
    context = nullcontext if optimizer else torch.no_grad
    batch: BatchEncoding
    overall_loss = 0
    count_items = 1
    for count_items, batch in get_iterable(
        enumerate(loader, 1),
        total=len(loader),
        is_main=accelerator.is_main_process,
        leave=False,
    ):
        if optimizer:
            optimizer.zero_grad()
        images = batch.pop("images")
        with context():
            outputs = model(images)
        with torch.no_grad():
            targets = text_model(**batch)
        loss = criterion(outputs, targets)
        overall_loss += loss.item()
        if optimizer:
            accelerator.backward(loss)
            optimizer.step()
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(loss)
    return overall_loss / count_items


def train_clip_model(
    accelerator: Accelerator,
    n_epochs: int,
    clip: CLIP,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    scheduler: Optional[Scheduler] = None,
) -> CLIP:
    for i in get_iterable(range(1, n_epochs + 1), total=n_epochs, is_main=accelerator.is_main_process):
        train_loss = run_clip_model(
            accelerator=accelerator,
            clip=clip,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        valid_loss = run_clip_model(accelerator=accelerator, clip=clip, loader=valid_loader, criterion=criterion)
        if accelerator.is_main_process:
            print(f"Epoch: {i:<3}")
            print(f"Train loss: {train_loss:.2f}")
            print(f"Validation loss: {valid_loss:.2f}")
    test_loss = run_clip_model(accelerator=accelerator, clip=clip, loader=test_loader, criterion=criterion)
    if accelerator.is_main_process:
        print(f"Test loss: {test_loss:.2f}")
    return clip


def run_clip_model(
    accelerator: Accelerator,
    clip: CLIP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Scheduler] = None,
) -> float:
    context = nullcontext if optimizer else torch.no_grad
    batch: BatchEncoding
    overall_loss = 0
    count_items = 1
    for count_items, batch in get_iterable(
        enumerate(loader, 1),
        total=len(loader),
        is_main=accelerator.is_main_process,
        leave=False,
    ):
        if optimizer:
            optimizer.zero_grad()
        images = batch.pop("images")
        with context():
            outputs = clip.forward(images=images, **batch)
        targets = torch.diag(torch.ones(images.size(0))).to(accelerator.device)
        loss = criterion(outputs.logits.image, targets) + criterion(outputs.logits.text, targets)
        overall_loss += loss.item()
        if optimizer:
            accelerator.backward(loss)
            optimizer.step()
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
    if scheduler and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(loss)
    return overall_loss / count_items
