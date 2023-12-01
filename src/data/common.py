from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

import albumentations as A
from PIL.Image import Image
from transformers import BatchEncoding

from src.types import Processor, Tokenizer

TOKENIZER_KWARGS = {"padding": True, "truncation": True, "return_tensors": "pt", "return_token_type_ids": False}


def create_transform(p: float = 0.5, train: bool = False) -> A.Compose:
    assert 0 < p < 1, "Probability must be in (0, 1)"
    transforms = [
        A.CLAHE(),
        A.GaussNoise(),
        A.CoarseDropout(),
        A.ChannelDropout(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    ]
    if train:
        return A.Compose([A.HorizontalFlip(p=p), A.OneOf(transforms=transforms, p=p)])
    return A.HorizontalFlip(p=p)


def collate_fn(
    items: Iterable[Tuple[Image, str]], processor: Processor, tokenizer: Tokenizer, max_length: Optional[int]
) -> BatchEncoding:
    images: List[Image] = []
    texts: List[str] = []
    for image, text in items:
        images.append(image)
        texts.append(text)
    batch = tokenizer(text=texts, max_length=max_length, **TOKENIZER_KWARGS)
    img_batch = processor(images=images, return_tensors="pt")
    assert len(img_batch) and "pixel_values" in img_batch, img_batch.keys()
    batch.update(img_batch)
    return batch


def create_collate_fn(
    processor: Processor, tokenizer: Tokenizer, max_length: Optional[int]
) -> Callable[[Iterable[Tuple[Image, str]]], BatchEncoding]:
    return partial(collate_fn, processor=processor, tokenizer=tokenizer, max_length=max_length)


def collate_distil_fn(  # pylint: disable=too-many-locals
    items: Iterable[Tuple[Image, str]],
    processor: Processor,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    teacher_processor: Processor,
    teacher_tokenizer: Tokenizer,
    teacher_max_length: Optional[int],
) -> BatchEncoding:
    images: List[Image] = []
    texts: List[str] = []
    for image, text in items:
        images.append(image)
        texts.append(text)

    batch = tokenizer(text=texts, max_length=max_length, **TOKENIZER_KWARGS)
    img_batch = processor(images=images, return_tensors="pt")
    assert len(img_batch) and "pixel_values" in img_batch, img_batch.keys()
    batch.update(img_batch)

    teacher_batch = teacher_tokenizer(text=texts, max_length=teacher_max_length, **TOKENIZER_KWARGS)
    teacher_img_batch = teacher_processor(images=images, return_tensors="pt")
    assert len(teacher_img_batch) and "pixel_values" in teacher_img_batch, teacher_img_batch.keys()
    teacher_batch.update(teacher_img_batch)

    for key, value in teacher_batch.items():
        batch["tchr_" + key] = value
    return batch


def create_distil_collate_fn(
    processor: Processor,
    tokenizer: Tokenizer,
    max_length: Optional[int],
    teacher_processor: Processor,
    teacher_tokenizer: Tokenizer,
    teacher_max_length: Optional[int],
) -> Callable[[Iterable[Tuple[Image, str]]], BatchEncoding]:
    return partial(
        collate_distil_fn,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        teacher_processor=teacher_processor,
        teacher_tokenizer=teacher_tokenizer,
        teacher_max_length=teacher_max_length,
    )
