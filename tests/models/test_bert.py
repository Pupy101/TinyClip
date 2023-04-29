from itertools import product
from typing import Any, Dict, Union

import pytest
import torch
from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel

from clip.models import create_bert, create_distil_bert
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
        "position_embedding_type",
    ],
    product(
        [TextModelType.BERT.value, TextModelType.DISTILBERT.value],
        [1_000],
        [256, 512],
        [2, 4],
        [4, 8],
        [1024, 2048],
        ["gelu_new"],
        [0.1],
        [0.1],
        [256],
        ["relative_key_query"],
    ),
)
def test_custom_bert(  # pylint: disable=too-many-locals
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
    position_embedding_type: str,
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
        "position_embedding_type": position_embedding_type,
    }
    config: Union[BertConfig, DistilBertConfig]
    model: Union[BertModel, DistilBertModel]
    if model_type == TextModelType.BERT.value:
        config, model = create_bert(**kwargs)
    elif model_type == TextModelType.DISTILBERT.value:
        config, model = create_distil_bert(**kwargs)
    else:
        raise RuntimeError
    _ = model.to(device)
    input_ids = torch.arange(128, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(device)
    with torch.no_grad():
        output = model.forward(input_ids)
    assert tuple(output.last_hidden_state.shape) == (batch_size, input_ids.shape[1], config.hidden_size)
