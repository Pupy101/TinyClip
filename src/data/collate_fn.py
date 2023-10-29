from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

from PIL.Image import Image
from torchvision.transforms import v2
from transformers import BatchEncoding

from src.types import Processor, Tokenizer

TOKENIZER_KWARGS = {"padding": True, "truncation": True, "return_tensors": "pt", "return_token_type_ids": False}


def create_transform() -> v2.Compose:
    return v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomAutocontrast(p=0.3), v2.RandomEqualize(p=0.3)])


def collate_fn(
    items: Iterable[Tuple[Image, str]],
    processor: Processor,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    transform: Optional[v2.Compose],
) -> BatchEncoding:
    images: List[Image] = []
    texts: List[str] = []
    for image, text in items:
        images.append(image if transform is None else transform(image))
        texts.append(text)
    batch = tokenizer(text=texts, max_length=max_length, **TOKENIZER_KWARGS)
    batch.update(processor(images=images, return_tensors="pt"))
    return batch


def create_collate_fn(
    processor: Processor,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    transform: Optional[v2.Compose],
) -> Callable[[Iterable[Tuple[Image, str]]], BatchEncoding]:
    return partial(collate_fn, processor=processor, tokenizer=tokenizer, max_length=max_length, transform=transform)


def collate_distil_fn(
    items: Iterable[Tuple[Image, str]],
    processor: Processor,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    teacher_processor: Processor,
    teacher_tokenizer: Tokenizer,
    teacher_max_length: Optional[int],
    transform: Optional[v2.Compose],
) -> BatchEncoding:
    images: List[Image] = []
    texts: List[str] = []
    for image, text in items:
        images.append(image if transform is None else transform(image))
        texts.append(text)

    batch = tokenizer(text=texts, max_length=max_length, **TOKENIZER_KWARGS)
    batch.update(processor(images=images, return_tensors="pt"))

    teacher_batch = teacher_tokenizer(text=texts, max_length=teacher_max_length, **TOKENIZER_KWARGS)
    teacher_batch.update(teacher_processor(images=images, return_tensors="pt"))

    for key, value in teacher_batch.items():
        batch["teacher_" + key] = value
    return batch


def create_distil_collate_fn(
    processor: Processor,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    teacher_processor: Processor,
    teacher_tokenizer: Tokenizer,
    teacher_max_length: Optional[int],
    transform: Optional[v2.Compose],
) -> Callable[[Iterable[Tuple[Image, str]]], BatchEncoding]:
    return partial(
        collate_distil_fn,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        teacher_processor=teacher_processor,
        teacher_tokenizer=teacher_tokenizer,
        teacher_max_length=teacher_max_length,
        transform=transform,
    )
