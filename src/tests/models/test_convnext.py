import torch

from src.models import ConvNeXt
from src.types import ConvNeXtConfig

default = ConvNeXtConfig(
    in_channels=3,
    out_channels=256,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)


def test_base_convnext():
    batch_size = 2
    input_tensor = torch.rand(batch_size, 3, 224, 224)
    model = ConvNeXt(config=default)
    with torch.no_grad():
        output = model.forward(input_tensor)
    assert tuple(output.shape) == (batch_size, model.config.out_channels)


def test_custom_convnext():
    batch_size = 2
    input_tensor = torch.rand(batch_size, 3, 224, 224)
    config = ConvNeXtConfig(
        in_channels=3,
        out_channels=1024,
        drop_path_rate=0.1,
        depths=[3, 3, 3, 9, 3],
        dims=[64, 256, 512, 1024, 2048],
    )
    model = ConvNeXt(config)
    with torch.no_grad():
        output = model.forward(input_tensor)
    assert tuple(output.shape) == (batch_size, model.config.out_channels)
