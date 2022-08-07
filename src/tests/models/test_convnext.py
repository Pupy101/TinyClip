import pytest
import torch

from src.models import ConvNeXt
from src.types import ConvNeXtConfig

default_config = ConvNeXtConfig(
    in_channels=3,
    out_channels=256,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)
custom_config = ConvNeXtConfig(
    in_channels=3,
    out_channels=1024,
    drop_path_rate=0.1,
    depths=[3, 3, 3, 9, 3],
    dims=[64, 256, 512, 1024, 2048],
)


@pytest.mark.parametrize(["config"], [(default_config,), (custom_config,)])
def test_base_convnext(config):
    batch_size = 2
    input_tensor = torch.rand(batch_size, 3, 224, 224)
    model = ConvNeXt(config=config)
    with torch.no_grad():
        output = model.forward(input_tensor)
    assert tuple(output.shape) == (batch_size, config.out_channels)
