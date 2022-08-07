import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.types import MultiTaskProportion, TrainConfig, MultiTaskDataLoaders

from ..models.clip import CLIP
from .engine import Engine

logger = logging.getLogger(__file__)

__DATE_FORMAT = "%Y_%m_%b_%H_%M_%S"


def train(config: TrainConfig, clip: CLIP, loaders: MultiTaskDataLoaders) -> None:
    """Function for training model with the specified parameters."""

    save_dir = Path(config.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    engine = Engine(
        seed=config.seed,
        clip=clip,
        criterion_clip=config.criterion_clip,
        criterion_image=config.criterion_image,
        criterion_text=config.criterion_text,
        optimizer=config.optimizer,
        device=config.device,
        scheduler=config.scheduler,
        count_accumulated_batches=config.count_accumulated_batches,
    )

    min_val_loss = float("inf")
    best_epoch = 0

    logger.info("Start training...")

    for i in range(1, config.n_epochs + 1):
        logger.info("Epoch %s", i)
        train_loss = train_epoch(
            engine=engine,
            clip_loader=loaders.clip.train,
            image_loader=loaders.image.train,
            text_loader=loaders.text.train,
            coefficients=config.coefficients,
            train_clip=config.train_clip,
            train_image=config.train_image,
            train_text=config.train_text,
        )
        eval_loss = eval_epoch(
            engine=engine,
            clip_loader=loaders.clip.validation,
            image_loader=loaders.image.validation,
            text_loader=loaders.text.validation,
            coefficients=config.coefficients,
            train_clip=config.train_clip,
            train_image=config.train_image,
            train_text=config.train_text,
        )
        print(
            f"Epoch: {i} | Train Loss: {train_loss:.5f} | Valid Loss: {eval_loss:.5f} "
        )
        if eval_loss < min_val_loss and train_loss > eval_loss:
            min_val_loss = eval_loss
            best_epoch = i
            checkpoint_name = "CLIP_" + datetime.now().strftime(__DATE_FORMAT) + ".pt"
            torch.save(engine.clip.state_dict(), save_dir / checkpoint_name)

    logger.info("Best epoch: %s", best_epoch)
    logger.info("Validation loss: %s", min_val_loss)


def train_epoch(
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    coefficients: MultiTaskProportion,
    train_clip: bool,
    train_image: bool,
    train_text: bool,
) -> float:
    """Function for train model on one epoch."""

    logger.info("Start train epoch ...")

    length_clip = len(clip_loader) if train_clip else 0
    length_image = len(image_loader) if train_image else 0
    length_text = len(text_loader) if train_text else 0
    length = length_clip + length_image + length_text

    overall_epoch_loss = 0.0

    pbar = tqdm(total=length, desc="Batch: | Loss: ", leave=False)

    if train_clip:
        engine.train()
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
        logger.info("CLIP metrics:")
        logger.info(str(engine.clip_metrics))
        *_, overall_loss = engine.clip_metrics.overall()
        overall_epoch_loss += overall_loss

    if train_image:
        engine.train()
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
        logger.info("Image classification metrics:")
        logger.info(str(engine.image_metrics))
        *_, overall_loss = engine.image_metrics.overall()
        overall_epoch_loss += overall_loss

    if train_text:
        engine.train()
        for batch in text_loader:
            loss = engine.text_part_forward(batch)
            if loss is not None:
                loss *= coefficients.text
                loss.backward()
                engine.optimization_step()
                pbar.set_description_str(
                    f"Train Text Loss: {loss.item():.2f}",
                    refresh=True,
                )
            pbar.update(1)
        logger.info("Masked LM metrics:")
        logger.info(str(engine.text_metrics))
        *_, overall_loss = engine.text_metrics.overall()
        overall_epoch_loss += overall_loss

    return overall_epoch_loss


@torch.no_grad()
def eval_epoch(
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    coefficients: MultiTaskProportion,
    train_clip: bool,
    train_image: bool,
    train_text: bool,
) -> float:
    """Function for evaluation model on one epoch."""

    logger.info("Start evaluation epoch ...")

    length_clip = len(clip_loader) if train_clip else 0
    length_image = len(image_loader) if train_image else 0
    length_text = len(text_loader) if train_text else 0
    length = length_clip + length_image + length_text

    overall_epoch_loss = 0.0

    pbar = tqdm(total=length, desc="Batch: | Loss: ", leave=False)

    if train_clip:
        engine.eval()
        for batch in clip_loader:
            loss = engine.clip_forward(batch)
            if loss is not None:
                loss *= coefficients.clip
                pbar.set_description_str(
                    f"Valid CLIP Loss: {loss.item():.2f}", refresh=True
                )
            pbar.update(1)
        logger.info("CLIP metrics:")
        logger.info(str(engine.clip_metrics))
        *_, overall_loss = engine.clip_metrics.overall()
        overall_epoch_loss += overall_loss

    if train_image:
        engine.eval()
        for batch in image_loader:
            loss = engine.image_part_forward(batch)
            if loss is not None:
                loss *= coefficients.image
                pbar.set_description_str(
                    f"Valid Image Loss: {loss.item():.2f}", refresh=True
                )
            pbar.update(1)
        logger.info("Image classification metrics:")
        logger.info(str(engine.image_metrics))
        *_, overall_loss = engine.image_metrics.overall()
        overall_epoch_loss += overall_loss

    if train_text:
        engine.eval()
        for batch in text_loader:
            loss = engine.text_part_forward(batch)
            if loss is not None:
                loss *= coefficients.text
                pbar.set_description_str(
                    f"Valid Text Loss: {loss.item():.2f}",
                    refresh=True,
                )
            pbar.update(1)
        logger.info("Masked LM metrics:")
        logger.info(str(engine.text_metrics))
        *_, overall_loss = engine.text_metrics.overall()
        overall_epoch_loss += overall_loss

    return overall_epoch_loss
