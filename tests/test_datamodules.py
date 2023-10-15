from pathlib import Path

import pytest
import torch

from src.data.datamodule import CLIPDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_clip_datamodule(batch_size: int) -> None:
    path = "data/valid.tsv"
    assert Path(path).exists()

    dm = CLIPDataModule(
        train_path=path,
        val_path=path,
        test_path=path,
        tokenizer="cointegrated/rubert-tiny2",
        max_length=256,
        batch_size=batch_size,
        num_workers=4,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        crop=250,
        size=224,
    )
    dm.setup("fit")
    assert dm.data_train is not None and dm.data_val is not None and dm.data_test is None
    assert dm.train_dataloader() and dm.val_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images = batch.pop("images")
    texts = batch.pop("input_ids")
    assert images.size(0) == batch_size
    assert texts.size(0) == batch_size
    assert images.dtype == torch.float32
    assert texts.dtype == torch.int64
