from typing import Tuple

from transformers import DebertaConfig, DebertaModel, DebertaV2Config, DebertaV2Model


def create_deberta(
    vocab_size: int,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 256,
    hidden_act: str = "gelu_new",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    relative_attention: bool = False,
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


def pretrained_deberta(pretrained: str) -> Tuple[DebertaConfig, DebertaModel]:
    model = DebertaModel.from_pretrained(pretrained)
    return model.config, model


def create_deberta_v2(
    vocab_size: int,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 256,
    hidden_act: str = "gelu_new",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    relative_attention: bool = False,
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


def pretrained_deberta_v2(pretrained: str) -> Tuple[DebertaV2Config, DebertaV2Model]:
    model = DebertaV2Model.from_pretrained(pretrained)
    return model.config, model
