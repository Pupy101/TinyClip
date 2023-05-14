from contextlib import nullcontext
from itertools import product
from typing import Any, Dict

import pytest
import torch

from clip.models import CLIP, ImagePartCLIP, TextPartCLIP
from clip.types import Device, ImageModelType, TextModelType

CONVNEXT_KWARGS = {
    "model_type": ImageModelType.CONVNEXT.value,
    "count_classes": 1000,
    "depths": [3, 3, 3, 3],
    "hidden_sizes": [32, 64, 128, 256],
    "drop_path_rate": 0.5,
    "hidden_act": "gelu",
}
CONVNEXTV2_KWARGS = {
    "model_type": ImageModelType.CONVNEXTV2.value,
    "count_classes": 1000,
    "depths": [3, 3, 3, 3],
    "hidden_sizes": [32, 64, 128, 256],
    "drop_path_rate": 0.5,
    "hidden_act": "gelu",
}
SWIN_KWARGS = {
    "model_type": ImageModelType.SWIN.value,
    "count_classes": 1000,
    "depths": [2, 2, 2, 2],
    "num_heads": [2, 4, 4, 8],
    "embed_dim": 32,
    "drop_path_rate": 0.1,
    "hidden_act": "gelu",
}
SWINV2_KWARGS = {
    "model_type": ImageModelType.SWINV2.value,
    "count_classes": 1000,
    "depths": [2, 2, 4, 2],
    "num_heads": [3, 4, 8, 12],
    "embed_dim": 64,
    "drop_path_rate": 0.1,
    "hidden_act": "gelu_new",
}

BERT_KWARGS = {
    "model_type": TextModelType.BERT.value,
    "vocab_size": 1000,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 1024,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 256,
    "position_embedding_type": "relative_key_query",
}
DISTILBERT_KWARGS = {
    "model_type": TextModelType.BERT.value,
    "vocab_size": 1000,
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 256,
    "position_embedding_type": "relative_key_query",
}
DEBERTA_KWARGS = {
    "model_type": TextModelType.DEBERTA.value,
    "vocab_size": 1_000,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 1024,
    "hidden_act": "gelu_new",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 256,
    "relative_attention": False,
}
DEBERTAV2_KWARGS = {
    "model_type": TextModelType.DEBERTAV2.value,
    "vocab_size": 1_000,
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "hidden_act": "gelu_new",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "relative_attention": True,
}


@pytest.mark.parametrize(
    "image_kwargs, text_kwargs",
    product(
        [CONVNEXT_KWARGS, CONVNEXTV2_KWARGS, SWIN_KWARGS],
        [BERT_KWARGS, DISTILBERT_KWARGS, DEBERTA_KWARGS, DEBERTAV2_KWARGS],
    ),
)
def test_clip(image_kwargs: Dict[str, Any], text_kwargs: Dict[str, Any], device: Device) -> None:
    image_model = ImagePartCLIP(**image_kwargs)
    text_model = TextPartCLIP(**text_kwargs)
    raise_assert = image_model.out_shape != text_model.out_shape
    context = pytest.raises(Exception) if raise_assert else nullcontext()
    with context:
        clip = CLIP(image_model, text_model).to(device)
    if raise_assert:
        return
    image = torch.rand(4, 3, 224, 224).to(device)
    text = torch.arange(256, dtype=torch.long).reshape(4, -1).to(device)
    with torch.no_grad():
        output = clip.forward({"images": image}, {"input_ids": text})
    assert tuple(output.logits.image.shape) == (4, 4)
    assert tuple(output.logits.text.shape) == (4, 4)
    assert tuple(output.embeddings.image.shape) == tuple(output.embeddings.text.shape)
    with torch.no_grad():
        output_image = clip.image_part.forward(image, classification=True)
    assert tuple(output_image.shape) == (4, image_model.count_classes)
    with torch.no_grad():
        output_text = clip.text_part.forward(text, masked_lm=True)
    assert tuple(output_text.shape) == (4, text_model.vocab_size, text.size(1))
