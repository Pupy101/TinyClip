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

    length = len(clip_loader) + len(image_loader) + len(text_loader)

    pbar = tqdm(total=length, desc="Batch: | Loss: ", leave=False)

    for batch in clip_loader:

        loss = engine.clip_forward(batch)

        if loss is not None:

            loss *= coefficients.clip

            loss.backward()
            engine.optimization_step()

            pbar.set_description_str(
                f"Train CLIP Loss: {loss.item():.2f}", refresh=True
            )
        pbar.update(1)

    for batch in image_loader:

        loss = engine.image_part_forward(batch)

        if loss is not None:

            loss *= coefficients.image

            loss.backward()
            engine.optimization_step()

            pbar.set_description_str(
                f"Train Image Loss: {loss.item():.2f}", refresh=True
            )
        pbar.update(1)

    for batch in text_loader:

        loss = engine.text_part_forward(batch)

        if loss is not None:

            loss *= coefficients.image

            loss.backward()
            engine.optimization_step()

            pbar.set_description_str(
                f"Train Text Loss: {loss.item():.2f}",
                refresh=True,
            )
        pbar.update(1)

    logger.info("CLIP metrics:")
    logger.info(str(engine.clip_metrics))
    logger.info("Image classification metrics:")
    logger.info(str(engine.image_metrics))
    logger.info("Masked LM metrics:")
    logger.info(str(engine.text_metrics))

    *_, overall_loss = engine.clip_metrics.overall()

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

    length = len(clip_loader) + len(image_loader) + len(text_loader)

    pbar = tqdm(total=length, desc="Batch: | Loss: ", leave=False)

    for batch in clip_loader:

        loss = engine.clip_forward(batch)

        if loss is not None:

            loss *= coefficients.clip

            pbar.set_description_str(
                f"Train CLIP Loss: {loss.item():.2f}", refresh=True
            )
        pbar.update(1)

    for batch in image_loader:

        loss = engine.image_part_forward(batch)

        if loss is not None:

            loss *= coefficients.image

            pbar.set_description_str(
                f"Train Image Loss: {loss.item():.2f}", refresh=True
            )
        pbar.update(1)

    for batch in text_loader:

        loss = engine.text_part_forward(batch)

        if loss is not None:

            loss *= coefficients.image

            pbar.set_description_str(
                f"Train Text Loss: {loss.item():.2f}",
                refresh=True,
            )
        pbar.update(1)

    logger.info("CLIP metrics:")
    logger.info(str(engine.clip_metrics))
    logger.info("Image classification metrics:")
    logger.info(str(engine.image_metrics))
    logger.info("Masked LM metrics:")
    logger.info(str(engine.text_metrics))

    *_, overall_loss = engine.clip_metrics.overall()

    return overall_loss
