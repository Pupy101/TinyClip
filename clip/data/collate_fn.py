from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import BatchEncoding, DataCollatorForLanguageModeling

from clip.types import Tokenizer


def clip_collate_fn(
    items: Iterable[Tuple[Tensor, str]],
    tokenizer: Tokenizer,
    max_length: Optional[int],
) -> BatchEncoding:
    texts: List[str] = []
    images: List[Tensor] = []
    for image, text in items:
        images.append(image.unsqueeze(0))
        texts.append(text)
    batched_images = torch.cat(images, dim=0)
    encoded = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    encoded["images"] = batched_images
    return encoded


def create_clip_collate_fn(
    tokenizer: Tokenizer,
    max_length: Optional[int] = None,
) -> Callable[[Iterable[Tuple[Tensor, str]]], BatchEncoding]:
    return partial(clip_collate_fn, tokenizer=tokenizer, max_length=max_length)


def mlm_collate_fn(
    items: Iterable[str],
    tokenizer: Tokenizer,
    max_length: Optional[int],
    collate_fn: DataCollatorForLanguageModeling,
) -> BatchEncoding:
    tokens = tokenizer(
        items,
        return_token_type_ids=False,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = collate_fn(tokens["input_ids"], return_tensors="pt")
    encoded["attention_mask"] = torch.FloatTensor(tokens["attention_mask"])
    return encoded


def create_masked_lm_collate_fn(
    tokenizer: Tokenizer,
    max_length: Optional[int] = None,
) -> Callable[[Iterable[str]], BatchEncoding]:
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    return partial(mlm_collate_fn, tokenizer=tokenizer, max_length=max_length, collate_fn=collate_fn)


def image_collate_fn(
    items: Iterable[Tuple[Tensor, int]],
    tokenizer: Tokenizer,
    max_length: Optional[int],
) -> BatchEncoding:
    images, texts = [], []
    for image, text in items:
        images.append(image.unsqueeze(0))
        texts.append(text)
    batch = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    batch["images"] = torch.cat(images, dim=0)
    return batch


def create_image_collate_fn(
    tokenizer: Tokenizer,
    max_length: Optional[int],
) -> Callable[[Iterable[str]], BatchEncoding]:
    return partial(image_collate_fn, tokenizer=tokenizer, max_length=max_length)
