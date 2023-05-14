from typing import List, Tuple

from transformers import ConvNextConfig, ConvNextModel, ConvNextV2Config, ConvNextV2Model


def create_convnext(
    depths: List[int],
    hidden_sizes: List[int],
    drop_path_rate: float = 0.5,
    hidden_act: str = "gelu",
    num_channels: int = 3,
) -> Tuple[ConvNextConfig, ConvNextModel]:
    assert len(depths) > 1, "Count depths must be more 1"
    assert len(depths) == len(hidden_sizes), "Count depths must equal hidden_sizes"
    config = ConvNextConfig(
        num_channels=num_channels,
        num_stages=len(hidden_sizes),
        hidden_sizes=hidden_sizes,
        depths=depths,
        hidden_act=hidden_act,
        drop_path_rate=drop_path_rate,
    )
    model = ConvNextModel(config=config)
    return config, model


def pretrained_convnext(pretrained: str) -> Tuple[ConvNextConfig, ConvNextModel]:
    model = ConvNextModel.from_pretrained(pretrained)
    return model.config, model


def create_convnext_v2(
    drop_path_rate: float,
    depths: List[int],
    hidden_sizes: List[int],
    hidden_act: str = "gelu",
    num_channels: int = 3,
) -> Tuple[ConvNextV2Config, ConvNextV2Model]:
    assert len(depths) > 1, "Count depths must be more 1"
    assert len(depths) == len(hidden_sizes), "Count depths must equal hidden_sizes"
    config = ConvNextV2Config(
        num_channels=num_channels,
        num_stages=len(hidden_sizes),
        hidden_sizes=hidden_sizes,
        depths=depths,
        hidden_act=hidden_act,
        drop_path_rate=drop_path_rate,
    )
    model = ConvNextV2Model(config=config)
    return config, model


def pretrained_convnext_v2(pretrained: str) -> Tuple[ConvNextV2Config, ConvNextV2Model]:
    model = ConvNextV2Model.from_pretrained(pretrained)
    return model.config, model
