from typing import List, Tuple

from transformers import SwinConfig, SwinModel, Swinv2Config, Swinv2Model


def create_swin(
    depths: List[int],
    num_heads: List[int],
    num_channels: int = 3,
    embed_dim: int = 96,
    window_size: int = 7,
    mlp_ratio: float = 4,
    hidden_dropout_prob: float = 0.3,
    attention_probs_dropout_prob: float = 0.3,
    drop_path_rate: float = 0.1,
    hidden_act: str = "gelu",
    use_absolute_embeddings: bool = False,
) -> Tuple[SwinConfig, SwinModel]:
    assert len(depths) > 1, "Count depths must be more 1"
    assert len(depths) == len(num_heads), "Count depths must equal hidden_sizes"
    config = SwinConfig(
        num_channels=num_channels,
        depths=depths,
        num_heads=num_heads,
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


def pretrained_swin(pretrained: str) -> Tuple[SwinConfig, SwinModel]:
    model = SwinModel.from_pretrained(pretrained)
    return model.config, model


def create_swin_v2(
    depths: List[int],
    num_heads: List[int],
    num_channels: int = 3,
    embed_dim: int = 96,
    window_size: int = 7,
    mlp_ratio: float = 4,
    hidden_dropout_prob: float = 0,
    attention_probs_dropout_prob: float = 0,
    drop_path_rate: float = 0.1,
    hidden_act: str = "gelu",
    use_absolute_embeddings: bool = False,
) -> Tuple[Swinv2Config, Swinv2Model]:
    assert len(depths) > 1, "Count depths must be more 1"
    assert len(depths) == len(num_heads), "Count depths must equal hidden_sizes"
    config = Swinv2Config(
        num_channels=num_channels,
        depths=depths,
        num_heads=num_heads,
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


def pretrained_swin_v2(pretrained: str) -> Tuple[Swinv2Config, Swinv2Model]:
    model = Swinv2Model.from_pretrained(pretrained)
    return model.config, model
