from typing import Tuple

from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel


def create_bert(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    hidden_act: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    position_embedding_type: str,
) -> Tuple[BertConfig, BertModel]:
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        position_embedding_type=position_embedding_type,
    )
    model = BertModel(config=config)
    return config, model


def create_distil_bert(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    hidden_act: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    position_embedding_type: str,
) -> Tuple[DistilBertConfig, DistilBertModel]:
    config = DistilBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        position_embedding_type=position_embedding_type,
    )
    model = DistilBertModel(config=config)
    return config, model
