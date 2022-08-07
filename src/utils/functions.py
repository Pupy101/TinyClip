"""
Module with custom functions
"""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import Generator, Iterable, List, Tuple

import requests
import torch
from torch import Tensor, nn

from src.types import DownloadFile, Item


def normalize(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Function for normalize tensor along 1 dimension."""
    norm = torch.sqrt(torch.sum(tensor**2, dim=dim)).unsqueeze(1)
    return tensor / norm


def freeze_weight(model: nn.Module) -> None:
    """Function for freeze weight in model."""
    for weight in model.parameters():
        weight.requires_grad = False


def compute_f1_batch(logits: Tensor, target: Tensor) -> Tuple[float, float, float]:
    """Function for compute TP, FP, FN from logits."""
    predict = torch.softmax(logits.detach(), dim=-1)
    true_positive = predict * target
    false_positive = predict - true_positive
    false_negative = target - true_positive
    return (
        true_positive.sum().cpu().item(),
        false_positive.sum().cpu().item(),
        false_negative.sum().cpu().item(),
    )


def compute_accuracy_1(logits: Tensor, target: Tensor) -> int:
    """Function for compute right classificated classes."""
    output_labels = torch.argmax(logits.detach(), dim=-1)
    return round(torch.sum(output_labels == target).cpu().item())


def compute_accuracy_5(logits: Tensor, target: Tensor) -> int:
    """Function for compute right classificated top5 classes."""
    output_labels = torch.topk(logits.detach(), k=5, dim=-1).indices
    target = target.unsqueeze(-1)
    return round(torch.sum(output_labels == target).cpu().item())


def generator_chunks(
    items: Iterable[Item], chunk_size: int = 10
) -> Generator[List[Item], None, None]:
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def download_file(item: DownloadFile) -> None:
    response = requests.get(item.url)
    with open(item.file_path, "wb") as file:
        file.write(response.content)


def download(items: List[DownloadFile], max_workers: int = 20) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_file, items)


def download_files_mp(files: List[DownloadFile], n_pools: int = 4) -> None:
    chunks = list(generator_chunks(files))
    with Pool(n_pools) as pool:
        pool.map(download, chunks)
