from typing import Callable, Dict, List, Union

from torch import cat, Tensor


def collate_fabric(tokenizer: Callable, max_len: int = 24):

    def url_collate(batch: List[Dict[str, Union[None, Tensor]]]) -> Dict[str, Tensor]:
        images = []
        descriptions = []
        for pair in batch:
            image, text = pair['image'], pair['text']
            if image is None:
                continue
            image: Tensor
            images.append(image.unsqueeze(0))
            descriptions.append(text.unsqueeze(0))
        return {
            'image': cat(images, dim=0),
            'text': tokenizer(
                descriptions, padding=True, truncation=True, max_length=max_len, return_tensors='pt'
            ),
        }
    return url_collate
