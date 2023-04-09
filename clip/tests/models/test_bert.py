import pytest
import torch

from clip.models import Bert, BertConfig

default_config = BertConfig(
    num_hidden_layers=4,
    vocab_size=8_000,
    hidden_size=256,
    num_attention_heads=4,
    intermediate_size=1024,
    position_embedding_type="relative_key_query",
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.5,
    hidden_act="gelu",
    max_pos_emb=512,
)
custom_config = BertConfig(
    num_hidden_layers=4,
    vocab_size=1_000,
    hidden_size=512,
    num_attention_heads=8,
    intermediate_size=2048,
    position_embedding_type="relative_key_query",
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.5,
    hidden_act="relu",
    max_position_embeddings=512,
)


@pytest.mark.parametrize(["config"], [(default_config,), (custom_config,)])
def test_deberta(config: BertConfig, device: torch.device):
    batch_size = 2
    model = Bert(config=config).to(device)
    x = torch.arange(512, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(device)
    with torch.no_grad():
        output = model.forward(x)
    assert tuple(output.shape) == (batch_size, x.shape[1], config.hidden_size)
