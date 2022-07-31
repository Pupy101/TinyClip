import logging
from datetime import datetime

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

    for i in range(1, parameters.n_epochs + 1):
        logger.info("Epoch %s", i)
        train_loss = train_epoch(
            engine=parameters.engine,
            clip_loader=parameters.dataloaders.clip.train,
            image_loader=parameters.dataloaders.image.train,
            text_loader=parameters.dataloaders.text.train,
            accumulation=parameters.accumulation_steps,
            coefficients=parameters.coefficients,
        )
        eval_loss = eval_epoch(
            engine=parameters.engine,
            clip_loader=parameters.dataloaders.clip.train,
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
    accumulation: int,
    coefficients: MultiTaskProportion,
) -> float:
    """Function for train model on one epoch."""
    engine.train()
    logger.info("Start train epoch ...")

    clip_iteration = True
    image_iteration = True
    text_iteration = True

    clip_iterator = iter(clip_loader)
    image_iterator = iter(image_loader)
    text_iterator = iter(text_loader)

    step_ids = 0
    length_loader = len(clip_loader)

    pbar = tqdm(total=length_loader, desc="Batch: | Loss: ", leave=False)

    while clip_iteration:
        batch = get_batch(clip_iterator)
        if batch is None:
            clip_iteration = False
            break
        loss = engine.clip_forward(batch) * coefficients.clip

        if image_iteration:
            batch = get_batch(image_iterator)
            if batch is None:
                image_iteration = False
                logger.info("Image loader empty on step: %s", step_ids + 1)
            else:
                loss += engine.image_part_forward(batch) * coefficients.image

        if text_iteration:
            batch = get_batch(text_iterator)
            if batch is None:
                text_iteration = False
                logger.info("Text loader empty on step: %s", step_ids + 1)
            else:
                loss += engine.text_part_forward(batch) * coefficients.text

        loss.backward()

        if ((step_ids + 1) % accumulation == 0) or (step_ids + 1 == length_loader):
            engine.optimization_step()

        step_ids += 1
        pbar_desc = f"Batch: {step_ids} | Loss: {loss.item():.2f} "
        pbar.set_description_str(pbar_desc, refresh=True)
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

    *_, overall_loss = clip_metrics.overall()

    return overall_loss


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

    clip_iteration = True
    image_iteration = True
    text_iteration = True

    clip_iterator = iter(clip_loader)
    image_iterator = iter(image_loader)
    text_iterator = iter(text_loader)

    step_ids = 0
    length_loader = len(clip_loader)

    pbar = tqdm(total=length_loader, desc="Batch: | Loss: ", leave=False)

    while clip_iteration:
        batch = get_batch(clip_iterator)
        if batch is None:
            clip_iteration = False
            break
        loss = engine.clip_forward(batch) * coefficients.clip

        if image_iteration:
            batch = get_batch(image_iterator)
            if batch is None:
                image_iteration = False
                logger.info("Image loader empty on step: %s", step_ids + 1)
            else:
                loss += engine.image_part_forward(batch) * coefficients.image

        if text_iteration:
            batch = get_batch(text_iterator)
            if batch is None:
                text_iteration = False
                logger.info("Text loader empty on step: %s", step_ids + 1)
            else:
                loss += engine.text_part_forward(batch) * coefficients.text

        step_ids += 1
        pbar_desc = f"Batch: {step_ids} | Loss: {loss.item():.2f} "
        pbar.set_description_str(pbar_desc, refresh=True)
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

    *_, overall_loss = clip_metrics.overall()

    return overall_loss
