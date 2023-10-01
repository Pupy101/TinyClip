from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

from torch import LongTensor, Tensor, cat
from transformers import BatchEncoding

from src.types import Tokenizer


def collate_fn(
    items: Iterable[Tuple[Tensor, Optional[str], Optional[str]]],
    tokenizer: Tokenizer,
    max_length: Optional[int],
) -> BatchEncoding:
    image_indexes: List[int] = []
    text_indexes: List[int] = []
    texts: List[str] = []
    images: List[Tensor] = []
    counter = 0
    for i, (image, ru_text, en_text) in enumerate(items):
        images.append(image.unsqueeze(0))
        if ru_text is not None:
            texts.append(ru_text)
            image_indexes.append(i)
            text_indexes.append(counter)
            counter += 1
        if en_text is not None:
            texts.append(en_text)
            image_indexes.append(i)
            text_indexes.append(counter)
            counter += 1
    batched_images = cat(images, dim=0)
    encoded = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    encoded.update(
        {"images": batched_images, "image_indexes": LongTensor(image_indexes), "text_indexes": LongTensor(text_indexes)}
    )
    return encoded


def create_collate_fn(
    tokenizer: Tokenizer,
    max_length: Optional[int] = None,
) -> Callable[[Iterable[Tuple[Tensor, str]]], BatchEncoding]:
    return partial(collate_fn, tokenizer=tokenizer, max_length=max_length)
