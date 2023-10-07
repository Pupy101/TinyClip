from pathlib import Path

import pytest
import torch

from src.data.datamodule import CLIPDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    path = "data/valid.tsv"
    assert Path(path).exists()

    dm = CLIPDataModule(
        train_path=path,
        val_path=path,
        test_path=path,
        tokenizer="cointegrated/rubert-tiny2",
        max_length=2048,
        batch_size=4,
        num_workers=4,
    )
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images = batch.pop("images")
    texts = batch.pop("input_ids")
    assert images.size(0) == batch_size
    assert texts.size(0) == batch_size
    assert images.dtype == torch.float32
    assert texts.dtype == torch.int64
