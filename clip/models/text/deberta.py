from typing import Tuple

from transformers import DebertaConfig, DebertaModel, DebertaV2Config, DebertaV2Model


def create_deberta(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    hidden_act: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    relative_attention: bool,
) -> Tuple[DebertaConfig, DebertaModel]:
    config = DebertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        relative_attention=relative_attention,
    )
    model = DebertaModel(config=config)
    return config, model


def create_deberta_v2(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    hidden_act: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    relative_attention: bool,
) -> Tuple[DebertaV2Config, DebertaV2Model]:
    config = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        relative_attention=relative_attention,
    )
    model = DebertaV2Model(config=config)
    return config, model
