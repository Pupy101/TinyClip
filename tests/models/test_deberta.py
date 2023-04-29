from itertools import product
from typing import Any, Dict, Union

import pytest
import torch
from transformers import DebertaConfig, DebertaModel, DebertaV2Config, DebertaV2Model

from clip.models import create_deberta, create_deberta_v2
from clip.types import TextModelType


@pytest.mark.parametrize(
    [
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "hidden_act",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "max_position_embeddings",
        "relative_attention",
    ],
    product(
        [TextModelType.DEBERTA.value, TextModelType.DEBERTAV2.value],
        [1_000],
        [256, 512],
        [2, 4],
        [4, 8],
        [1024, 2048],
        ["gelu_new"],
        [0.1],
        [0.1],
        [256],
        [True, False],
    ),
)
def test_custom_deberta(  # pylint: disable=too-many-locals
    model_type: str,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    hidden_act: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_position_embeddings: int,
    relative_attention: bool,
    device: torch.device,
) -> None:
    batch_size = 2
    kwargs: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "hidden_act": hidden_act,
        "hidden_dropout_prob": hidden_dropout_prob,
        "attention_probs_dropout_prob": attention_probs_dropout_prob,
        "max_position_embeddings": max_position_embeddings,
        "relative_attention": relative_attention,
    }
    config: Union[DebertaConfig, DebertaV2Config]
    model: Union[DebertaModel, DebertaV2Model]
    if model_type == TextModelType.DEBERTA.value:
        config, model = create_deberta(**kwargs)
    elif model_type == TextModelType.DEBERTAV2.value:
        config, model = create_deberta_v2(**kwargs)
    else:
        raise RuntimeError
    _ = model.to(device)
    input_ids = torch.arange(128, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(device)
    with torch.no_grad():
        output = model.forward(input_ids)
    assert tuple(output.last_hidden_state.shape) == (batch_size, input_ids.shape[1], config.hidden_size)
