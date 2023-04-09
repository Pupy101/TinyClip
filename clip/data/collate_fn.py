from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import BatchEncoding, DataCollatorForLanguageModeling

from clip.types import Tokenizer


def create_clip_collate_fn(
    tokenizer: Tokenizer, max_length: Optional[int] = None
) -> Callable[[Iterable[Tuple[Tensor, str]]], BatchEncoding]:
    def clip_collate_fn(items: Iterable[Tuple[Tensor, str]]) -> BatchEncoding:
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
        )
        encoded["image"] = batched_images
        return encoded

    return clip_collate_fn


def create_masked_lm_collate_fn(
    tokenizer: Tokenizer, max_length: Optional[int] = None
) -> Callable[[Iterable[str]], BatchEncoding]:
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    def mlm_collate_fn(items: Iterable[str]) -> BatchEncoding:
        input_ids = tokenizer(
            items,
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        print(input_ids)
        encoded = collate_fn(input_ids["input_ids"], return_tensors="pt")
        encoded["attention_mask"] = torch.FloatTensor(input_ids["attention_mask"])
        return encoded

    return mlm_collate_fn
