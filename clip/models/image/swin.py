from typing import List, Tuple

from transformers import SwinConfig, SwinModel, Swinv2Config, Swinv2Model


def create_swin(
    depths: List[int],
    num_heads: List[int],
    num_channels: int,
    embed_dim: int,
    window_size: int,
    mlp_ratio: float,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    drop_path_rate: float,
    hidden_act: str,
    use_absolute_embeddings: bool,
) -> Tuple[SwinConfig, SwinModel]:
    assert len(depths) > 1, "Count depths must be more 1"
    assert len(depths) == len(num_heads), "Count depths must equal hidden_sizes"
    config = SwinConfig(
        num_channels=num_channels,
        depths=list(depths),
        num_heads=list(num_heads),
        embed_dim=embed_dim,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        drop_path_rate=drop_path_rate,
        hidden_act=hidden_act,
        use_absolute_embeddings=use_absolute_embeddings,
    )
    model = SwinModel(config=config)
    return config, model


def create_swin_v2(
    depths: List[int],
    num_heads: List[int],
    num_channels: int,
    embed_dim: int,
    window_size: int,
    mlp_ratio: float,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    drop_path_rate: float,
    hidden_act: str,
    use_absolute_embeddings: bool,
) -> Tuple[Swinv2Config, Swinv2Model]:
    assert len(depths) > 1, "Count depths must be more 1"
    assert len(depths) == len(num_heads), "Count depths must equal hidden_sizes"
    config = Swinv2Config(
        num_channels=num_channels,
        depths=list(depths),
        num_heads=list(num_heads),
        embed_dim=embed_dim,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        drop_path_rate=drop_path_rate,
        hidden_act=hidden_act,
        use_absolute_embeddings=use_absolute_embeddings,
    )
    model = Swinv2Model(config=config)
    return config, model
