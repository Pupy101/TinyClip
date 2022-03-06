from typing import Dict, List, Union

from torch import cat, Tensor


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
        'text': cat(descriptions, dim=0),
    }
