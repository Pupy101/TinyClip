import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip.types import MultiTaskDataLoaders, PathLike, RunType
from clip.utils.function import zip_dataloaders

from .engine import Engine

logger = logging.getLogger(__file__)

DATE_FORMAT = "%Y_%m_%b"


def train(engine: Engine, loaders: MultiTaskDataLoaders, n_epochs: int, save_dir: PathLike) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    min_val_loss = float("inf")
    best_epoch = 0

    logger.info("Start training...")

    for i in range(1, n_epochs + 1):
        logger.info("Epoch %s", i)
        train_loss = run_epoch(
            engine=engine,
            clip_loader=loaders.clip.train,
            image_loader=loaders.image.train,
            text_loader=loaders.text.train,
            run_type=RunType.TRAIN.value,
        )
        valid_loss = run_epoch(
            engine=engine,
            clip_loader=loaders.clip.valid,
            image_loader=loaders.image.valid,
            text_loader=loaders.text.valid,
            run_type=RunType.VALID.value,
        )
        test_loss = run_epoch(
            engine=engine,
            clip_loader=loaders.clip.test,
            image_loader=loaders.image.test,
            text_loader=loaders.text.test,
            run_type=RunType.TEST.value,
        )
        print(
            f"Epoch: {i} | Train Loss: {train_loss:.5f} | " f"Valid Loss: {valid_loss:.5f} | Test Loss: {test_loss:.5f}"
        )
        if valid_loss < min_val_loss and train_loss > valid_loss:
            min_val_loss = test_loss
            best_epoch = i
            checkpoint_name = f"CLIP_{i}_" + datetime.now().date().strftime(DATE_FORMAT) + ".pt"
            torch.save(engine.clip.state_dict(), save_dir / checkpoint_name)

    logger.info("Best epoch: %s", best_epoch)
    logger.info("Validation loss: %s", min_val_loss)


def run_epoch(  # pylint: disable=too-many-locals
    engine: Engine,
    clip_loader: DataLoader,
    image_loader: DataLoader,
    text_loader: DataLoader,
    run_type: str,
) -> float:
    assert run_type in {_.value for _ in RunType}

    if run_type == RunType.TRAIN.value:
        engine.train()
        context = nullcontext()
    else:
        engine.eval()
        context = torch.no_grad()

    overall_epoch_loss = 0.0
    pbar = tqdm(total=len(clip_loader), desc="Batch: | Loss: ", leave=False)

    for clip_batch, image_batch, text_batch in zip_dataloaders(clip_loader, image_loader, text_loader):
        loss: torch.Tensor = torch.tensor(0.0).to(engine.device)

        with context:
            loss_clip = engine.clip_forward(clip_batch)

        if loss_clip is not None:
            loss += loss_clip

        if image_batch:
            with context:
                loss += engine.image_part_forward(image_batch)

        if text_batch:
            with context:
                loss += engine.text_part_forward(text_batch)

        if loss.item() == 0:
            continue

        if run_type == RunType.TRAIN.value:
            loss.backward()
            engine.optimization_step()

        overall_epoch_loss += loss.item()

        pbar.set_description_str(f"Loss: {loss.item():.2f}", refresh=True)
        pbar.update(1)

    logger.info("CLIP metrics:")
    logger.info(str(engine.clip_metrics))

    logger.info("Image classification metrics:")
    logger.info(engine.image_metrics)

    logger.info("Masked LM metrics:")
    logger.info(str(engine.text_metrics))

    return overall_epoch_loss / len(clip_loader)
