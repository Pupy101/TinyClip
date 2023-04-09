from functools import partial
from multiprocessing import Pool
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from clip.types import PathLike


def freeze_weight(model: nn.Module) -> None:
    """Function for freeze weight in model"""

    for weight in model.parameters():
        weight.requires_grad = False


def count_params(model: nn.Module, requires_grad: Optional[bool] = None) -> int:
    count = 0
    for params in model.parameters():
        if requires_grad is not None and params.requires_grad != requires_grad:
            continue
        count += params.numel()
    return count


def resize_image(path: PathLike, size: int) -> None:
    path = str(path)
    image = cv2.imread(path)
    height, width, _ = image.shape
    if height <= width:
        height, width = size, round(size * width / height)
    else:
        height, width = round(size * height / width), size
    resized = cv2.resize(image, (width, height))
    cv2.imwrite(path, resized)


def resize_image_mp(pathes: Iterable[PathLike], size: int, n_pools: int, tqdm_off: bool) -> None:
    resizer = partial(resize_image, size=size)
    with Pool(n_pools) as pool:
        list(tqdm(pool.map(resizer, pathes), disable=tqdm_off))


def split_train_val_test(
    dataframe: pd.DataFrame,
    train_size: float,
    valid_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = dataframe.sample(frac=1)
    assert train_size + valid_size < 1, "Sum of train/valid sizes must be smaller 1"
    train = round(dataframe.shape[0] * train_size)
    valid = round(dataframe.shape[0] * valid_size)
    train_df, valid_df, test_df = (
        shuffled[:train].reset_index(),
        shuffled[train : train + valid],
        shuffled[train + valid :],
    )
    train_df.reset_index(drop=True)
    valid_df.reset_index(drop=True)
    test_df.reset_index(drop=True)
    return train_df, valid_df, test_df


def split_train_val_test_stratify(
    dataframe: pd.DataFrame,
    train_size: float,
    valid_size: float,
    stratify_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert train_size + valid_size < 1, "Sum of train/valid sizes must be smaller 1"
    unique_targets = dataframe[stratify_column].unique()

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for target in unique_targets:
        indices = dataframe[dataframe[stratify_column] == target].index
        np.random.shuffle(indices)

        split_train_size = int(train_size * len(indices))
        split_valid_size = int(train_size * len(indices))

        train_indices = indices[:split_train_size]
        valid_indices = indices[split_train_size : split_train_size + split_valid_size]
        test_indices = indices[split_train_size + split_valid_size :]

        train_df = pd.concat([train_df, dataframe.loc[train_indices]])
        valid_df = pd.concat([valid_df, dataframe.loc[valid_indices]])
        test_df = pd.concat([test_df, dataframe.loc[test_indices]])

    # Shuffle the rows of the resulting dataframes
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    return train_df, valid_df, test_df
