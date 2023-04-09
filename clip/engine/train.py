import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip.types import MultiTaskDataLoaders, MultiTaskProportions, TrainConfig

from ..models.clip import CLIP
from .engine import Engine

logger = logging.getLogger(__file__)

DATE_FORMAT = "%Y_%m_%b"


def train(config: TrainConfig, clip: CLIP, loaders: MultiTaskDataLoaders) -> None:
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        clip=clip,
        criterion=config.criterion,
        optimizer=config.optimizer,
        device=config.device,
        scheduler=config.scheduler,
        count_accumukating_steps=config.count_accumukating_steps,
        seed=config.seed,
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
        )
        valid_loss = eval_epoch(
            engine=engine,
            clip_loader=loaders.clip.valid,
            image_loader=loaders.image.valid,
            text_loader=loaders.text.valid,
            coefficients=config.coefficients,
        )
        test_loss = eval_epoch(
            engine=engine,
            clip_loader=loaders.clip.test,
            image_loader=loaders.image.test,
            text_loader=loaders.text.test,
            coefficients=config.coefficients,
        )
        print(
            f"Epoch: {i} | Train Loss: {train_loss:.5f} | "
            f"Valid Loss: {valid_loss:.5f} | Test Loss: {test_loss:.5f}"
        )
        if valid_loss < min_val_loss and train_loss > valid_loss:
            min_val_loss = test_loss
            best_epoch = i
            checkpoint_name = f"CLIP_{i}_" + datetime.now().date().strftime(DATE_FORMAT) + ".pt"
            torch.save(engine.clip.state_dict(), save_dir / checkpoint_name)

    logger.info("Best epoch: %s", best_epoch)
    logger.info("Validation loss: %s", min_val_loss)


def train_epoch(  # pylint: disable=too-many-locals
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    coefficients: MultiTaskProportions,
) -> float:
    run_clip, run_image, run_text = True, True, True
    clip_it, image_it, text_it = iter(clip_loader), iter(image_loader), iter(text_loader)

    overall_epoch_loss = 0.0

    pbar = tqdm(total=len(clip_loader), desc="Batch: | Loss: ", leave=False)

    engine.train()

    while run_clip:
        loss: torch.Tensor = torch.tensor(0)
        try:
            batch = next(clip_it)
        except StopIteration:
            run_clip = False
            continue
        loss_clip = engine.clip_forward(batch)
        if loss_clip is not None:
            loss += coefficients.clip * loss_clip

        if run_image:
            try:
                batch = next(image_it)
            except StopIteration:
                run_image = False
            if run_image:
                loss_image = engine.image_part_forward(batch)
                if loss_image is not None:
                    loss += coefficients.image * loss_image

        if run_text:
            try:
                batch = next(text_it)
            except StopIteration:
                run_text = False
            if run_text:
                loss_text = engine.text_part_forward(batch)
                if loss_text is not None:
                    loss += coefficients.text * loss_text
        if loss == 0:
            continue
        loss.backward()
        engine.optimization_step()
        pbar.set_description_str(f"Loss: {loss.item():.2f}", refresh=True)
        overall_epoch_loss += loss.item()
        pbar.update(1)

    logger.info("CLIP metrics:")
    logger.info(str(engine.clip_metrics))

    logger.info("Image classification metrics:")
    logger.info(str(engine.image_metrics))

    logger.info("Masked LM metrics:")
    logger.info(str(engine.text_metrics))

    return overall_epoch_loss / len(clip_loader)


@torch.no_grad()
def eval_epoch(  # pylint: disable=too-many-locals
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    coefficients: MultiTaskProportions,
) -> float:
    run_clip, run_image, run_text = True, True, True
    clip_it, image_it, text_it = iter(clip_loader), iter(image_loader), iter(text_loader)

    overall_epoch_loss = 0.0

    pbar = tqdm(total=len(clip_loader), desc="Batch: | Loss: ", leave=False)

    engine.train()

    while run_clip:
        loss: torch.Tensor = torch.tensor(0)
        try:
            batch = next(clip_it)
        except StopIteration:
            run_clip = False
            continue
        loss_clip = engine.clip_forward(batch)
        if loss_clip is not None:
            loss += coefficients.clip * loss_clip

        if run_image:
            try:
                batch = next(image_it)
            except StopIteration:
                run_image = False
            if run_image:
                loss_image = engine.image_part_forward(batch)
                if loss_image is not None:
                    loss += coefficients.image * loss_image

        if run_text:
            try:
                batch = next(text_it)
            except StopIteration:
                run_text = False
            if run_text:
                loss_text = engine.text_part_forward(batch)
                if loss_text is not None:
                    loss += coefficients.text * loss_text
        if loss == 0:
            continue
        pbar.set_description_str(f"Loss: {loss.item():.2f}", refresh=True)
        overall_epoch_loss += loss.item()
        pbar.update(1)

    logger.info("CLIP metrics:")
    logger.info(str(engine.clip_metrics))

    logger.info("Image classification metrics:")
    logger.info(str(engine.image_metrics))

    logger.info("Masked LM metrics:")
    logger.info(str(engine.text_metrics))

    return overall_epoch_loss / len(clip_loader)
