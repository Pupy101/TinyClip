from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

from torch import LongTensor, Tensor, cat
from transformers import BatchEncoding

from src.types import Tokenizer

TOKENIZER_KWARGS = {"padding": True, "truncation": True, "return_tensors": "pt", "return_token_type_ids": False}


def collate_fn(
    items: Iterable[Tuple[Tensor, Union[str, List[str]], Optional[int]]],
    tokenizer: Tokenizer,
    max_length: Optional[int],
) -> BatchEncoding:
    batch_images: List[Tensor] = []
    batch_texts: List[str] = []
    batch_labels: List[int] = []

    for image, text, label in items:
        batch_images.append(image.unsqueeze(0))
        if isinstance(text, list) and not batch_texts:
            batch_texts = text
        elif isinstance(text, str):
            batch_texts.append(text)
        if label is not None:
            batch_labels.append(label)

    batch = tokenizer(text=batch_texts, max_length=max_length, **TOKENIZER_KWARGS)
    batch["image"] = cat(batch_images, dim=0)
    if batch_labels:
        batch["label"] = LongTensor(batch_labels)

    return batch


def create_collate_fn(
    tokenizer: Tokenizer, max_length: Optional[int] = None
) -> Callable[[Iterable[Tuple[Optional[Tensor], List[str]]]], BatchEncoding]:
    return partial(collate_fn, tokenizer=tokenizer, max_length=max_length)


def collate_with_teacher_fn(
    items: Iterable[Tuple[Tensor, Union[str, List[str]], Optional[int]]],
    tokenizer: Tokenizer,
    max_length: Optional[int],
    teacher_tokenizer: Tokenizer,
    teacher_max_length: Optional[int],
) -> BatchEncoding:
    batch_images: List[Tensor] = []
    batch_texts: List[str] = []
    batch_labels: List[int] = []

    for image, text, label in items:
        batch_images.append(image.unsqueeze(0))
        if isinstance(text, list) and not batch_texts:
            batch_texts = text
        elif isinstance(text, str):
            batch_texts.append(text)
        if label is not None:
            batch_labels.append(label)

    batch = tokenizer(text=batch_texts, max_length=max_length, **TOKENIZER_KWARGS)
    for key, value in teacher_tokenizer(text=batch_texts, max_length=teacher_max_length, **TOKENIZER_KWARGS).items():
        batch["teacher_" + key] = value
    batch["image"] = cat(batch_images, dim=0)
    if batch_labels:
        batch["label"] = LongTensor(batch_labels)

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
