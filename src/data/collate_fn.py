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
) -> Callable[[Iterable[Tuple[Tensor, str]]], BatchEncoding]:
    return partial(collate_fn, tokenizer=tokenizer, max_length=max_length)
