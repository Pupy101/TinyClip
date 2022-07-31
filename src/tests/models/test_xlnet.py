import torch

from src.models import XLNet
from src.types import XLNetConfig

default_config = XLNetConfig(
    num_layers=4,
    vocab_size=8_000,
    model_dim=256,
    num_heads=4,
    feedforward_dim=1024,
    dropout=0.5,
    activation="gelu",
    pad_idx=0,
)


def test_xlnet():
    batch_size = 2
    model = XLNet(config=default_config)
    x = torch.arange(512, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    with torch.no_grad():
        output = model.forward(x)
    assert tuple(output.shape) == (batch_size, x.shape[1], model.config.model_dim)


def test_custom_mask_xlnet():
    batch_size = 2
    config = XLNetConfig(
        num_layers=4,
        vocab_size=1_000,
        model_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        dropout=0.5,
        activation="relu",
        pad_idx=0,
    )
    model = XLNet(config=config)
    x = torch.arange(512, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    with torch.no_grad():
        output = model.forward(x)
    assert tuple(output.shape) == (batch_size, x.shape[1], model.config.model_dim)