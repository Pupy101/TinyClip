from functools import partial
from random import choice
from typing import Callable, Iterable, List, Optional, Tuple

from torch import Tensor, cat
from transformers import BatchEncoding

from src.types import Tokenizer


def collate_fn(
    items: Iterable[Tuple[Optional[Tensor], List[str]]], tokenizer: Tokenizer, max_length: Optional[int]
) -> BatchEncoding:
    batch_images: List[Tensor] = []
    batch_texts: List[str] = []

    for image, texts in items:
        if image is None or not texts:
            continue
        batch_images.append(image.unsqueeze(0))
        batch_texts.append(choice(texts))

    batch = tokenizer(
        text=batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    batch["images"] = cat(batch_images, dim=0)

    return batch


def create_collate_fn(
    tokenizer: Tokenizer, max_length: Optional[int] = None
) -> Callable[[Iterable[Tuple[Optional[Tensor], List[str]]]], BatchEncoding]:
    return partial(collate_fn, tokenizer=tokenizer, max_length=max_length)


def collate_with_teacher_fn(
    items: Iterable[Tuple[Optional[Tensor], List[str]]],
    tokenizer: Tokenizer,
    max_length: Optional[int],
    teacher_tokenizer: Tokenizer,
    teacher_max_length: Optional[int],
) -> BatchEncoding:
    batch_images: List[Tensor] = []
    batch_texts: List[str] = []

    for image, texts in items:
        if image is None or not texts:
            continue
        batch_images.append(image.unsqueeze(0))
        batch_texts.append(choice(texts))

    batch = tokenizer(
        text=batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    teacher_batch = teacher_tokenizer(
        text=batch_texts,
        padding=True,
        truncation=True,
        max_length=teacher_max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    for key, value in teacher_batch.items():
        batch["teacher_" + key] = value

    batch["images"] = cat(batch_images, dim=0)

    return batch


def create_collate_with_teacher_fn(
    tokenizer: Tokenizer,
    teacher_tokenizer: Tokenizer,
    max_length: Optional[int] = None,
    teacher_max_length: Optional[int] = None,
) -> Callable[[Iterable[Tuple[Optional[Tensor], List[str]]]], BatchEncoding]:
    return partial(
        collate_with_teacher_fn,
        tokenizer=tokenizer,
        max_length=max_length,
        teacher_tokenizer=teacher_tokenizer,
        teacher_max_length=teacher_max_length,
    )
