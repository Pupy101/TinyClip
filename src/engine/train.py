import logging
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.types import MultiTaskProportion, TrainingParameters
from src.utils.functions import get_batch

from .engine import Engine

logger = logging.getLogger(__file__)

__DATE_FORMAT = "%Y_%m_%b_%H_%M_%S"


def train(parameters: TrainingParameters) -> None:
    """Function for training model with the specified parameters."""

    if not parameters.save_dir.exists():
        parameters.save_dir.mkdir(parents=True)

    min_val_loss = float("inf")
    best_epoch = 0

    logger.info("Start training...")

    for i in range(1, parameters.n_epochs + 1):
        logger.info("Epoch %s", i)
        train_loss = train_epoch(
            engine=parameters.engine,
            clip_loader=parameters.dataloaders.clip.train,
            image_loader=parameters.dataloaders.image.train,
            text_loader=parameters.dataloaders.text.train,
            coefficients=parameters.coefficients,
        )
        eval_loss = eval_epoch(
            engine=parameters.engine,
            clip_loader=parameters.dataloaders.clip.validation,
            image_loader=parameters.dataloaders.image.train,
            text_loader=parameters.dataloaders.text.train,
            coefficients=parameters.coefficients,
        )
        print(
            f"Epoch: {i} | Train Loss: {train_loss:.5f} | Valid Loss: {eval_loss:.5f} "
        )
        if eval_loss < min_val_loss and train_loss > eval_loss:
            min_val_loss = eval_loss
            best_epoch = i
            checkpoint_name = "CLIP_" + datetime.now().strftime(__DATE_FORMAT) + ".pt"
            torch.save(
                parameters.engine.clip.state_dict(),
                parameters.save_dir / checkpoint_name,
            )

    logger.info("Best epoch: %s", best_epoch)
    logger.info("Validation loss: %s", min_val_loss)


def train_epoch(
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    coefficients: MultiTaskProportion,
) -> float:
    """Function for train model on one epoch."""
    engine.train()
    logger.info("Start train epoch ...")

    length_clip = len(clip_loader)

    image_iteration = True
    text_iteration = True

    clip_iterator = iter(clip_loader)
    image_iterator = iter(image_loader)
    text_iterator = iter(text_loader)

    step_ids = 0
    length_loader = len(clip_loader)

    pbar = tqdm(total=length_loader, desc="Batch: | Loss: ", leave=False)

    for _ in range(length_clip):
        overall_loss: Optional[torch.Tensor] = None

        batch = get_batch(clip_iterator)

        if batch is not None:
            loss = engine.clip_forward(batch)
            if loss is not None:
                overall_loss = loss * coefficients.clip

        batch = get_batch(image_iterator)

        if batch is None and image_iteration:
            image_iteration = False
            logger.info("Image loader empty on step: %s", step_ids + 1)
        elif batch is not None and image_iteration:
            loss = engine.image_part_forward(batch)
            if loss is not None and overall_loss is not None:
                overall_loss += loss * coefficients.image
            elif loss is not None:
                overall_loss = loss * coefficients.image

        batch = get_batch(text_iterator)

        if batch is None and text_iteration:
            text_iteration = False
            logger.info("Text loader empty on step: %s", step_ids + 1)
        elif batch is not None and text_iteration:
            loss = engine.text_part_forward(batch)
            if loss is not None and overall_loss is not None:
                overall_loss += loss * coefficients.text
            elif loss is not None:
                overall_loss = loss * coefficients.text

        step_ids += 1
        if overall_loss is not None:
            overall_loss.backward()
            pbar.set_description_str(
                f"Batch train: {step_ids} | Loss: {overall_loss.item():.2f}",
                refresh=True,
            )
        pbar.update(1)

    logger.info("Overall step count: %s", step_ids)
    clip_metrics = engine.clip_metrics
    logger.info("CLIP metrics:")
    logger.info(str(clip_metrics))
    image_metrics = engine.image_metrics
    logger.info("Image classification metrics:")
    logger.info(str(image_metrics))
    text_metrics = engine.text_metrics
    logger.info("Masked LM metrics:")
    logger.info(str(text_metrics))

    *_, overall_loss_epoch = clip_metrics.overall()

    return overall_loss_epoch


@torch.no_grad()
def eval_epoch(
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    coefficients: MultiTaskProportion,
) -> float:
    """Function for evaluation model on one epoch."""

    engine.eval()
    logger.info("Start evaluation epoch ...")

    length_clip = len(clip_loader)

    image_iteration = True
    text_iteration = True

    clip_iterator = iter(clip_loader)
    image_iterator = iter(image_loader)
    text_iterator = iter(text_loader)

    step_ids = 0
    length_loader = len(clip_loader)

    pbar = tqdm(total=length_loader, desc="Batch: | Loss: ", leave=False)

    for _ in range(length_clip):
        overall_loss: Optional[torch.Tensor] = None

        batch = get_batch(clip_iterator)

        if batch is not None:
            loss = engine.clip_forward(batch)
            if loss is not None:
                overall_loss = loss * coefficients.clip

        batch = get_batch(image_iterator)

        if batch is None and image_iteration:
            image_iteration = False
            logger.info("Image loader empty on step: %s", step_ids + 1)
        elif batch is not None and image_iteration:
            loss = engine.image_part_forward(batch)
            if loss is not None and overall_loss is not None:
                overall_loss += loss * coefficients.image
            elif loss is not None:
                overall_loss = loss * coefficients.image

        batch = get_batch(text_iterator)

        if batch is None and text_iteration:
            text_iteration = False
            logger.info("Text loader empty on step: %s", step_ids + 1)
        elif batch is not None and text_iteration:
            loss = engine.text_part_forward(batch)
            if loss is not None and overall_loss is not None:
                overall_loss += loss * coefficients.text
            elif loss is not None:
                overall_loss = loss * coefficients.text

        step_ids += 1
        if overall_loss is not None:
            pbar.set_description_str(
                f"Batch valid: {step_ids} | Loss: {overall_loss.item():.2f}",
                refresh=True,
            )
        pbar.update(1)

    logger.info("Overall step count: %s", step_ids)
    clip_metrics = engine.clip_metrics
    logger.info("CLIP metrics:")
    logger.info(str(clip_metrics))
    image_metrics = engine.image_metrics
    logger.info("Image classification metrics:")
    logger.info(str(image_metrics))
    text_metrics = engine.text_metrics
    logger.info("Masked LM metrics:")
    logger.info(str(text_metrics))

    *_, overall_loss_epoch = clip_metrics.overall()

    return overall_loss_epoch
