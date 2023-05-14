from typing import Tuple

from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel


def create_bert(
    vocab_size: int,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 256,
    hidden_act: str = "gelu_new",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    position_embedding_type: str = "relative_key_query",
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


def pretrained_bert(pretrained: str) -> Tuple[BertConfig, BertModel]:
    model = BertModel.from_pretrained(pretrained)
    return model.config, model


def create_distil_bert(
    vocab_size: int,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 256,
    hidden_act: str = "gelu_new",
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    position_embedding_type: str = "relative_key_query",
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


def pretrained_distil_bert(pretrained: str) -> Tuple[DistilBertConfig, DistilBertModel]:
    model = DistilBertModel.from_pretrained(pretrained)
    return model.config, model
