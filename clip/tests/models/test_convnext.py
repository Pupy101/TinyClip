import pytest
import torch

from clip.models import ConvNeXt
from clip.types import ConvNeXtConfig

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


@pytest.mark.parametrize(
    "config, use_dw_conv",
    [(default_config, True), (default_config, False), (custom_config, False)],
)
def test_base_convnext(config, use_dw_conv, device: torch.device):
    config.use_dw_conv = use_dw_conv
    batch_size = 2
    input_tensor = torch.rand(batch_size, 3, 224, 224).to(device)
    model = ConvNeXt(config=config).to(device)
    with torch.no_grad():
        output = model.forward(input_tensor)
    assert tuple(output.shape) == (batch_size, config.out_channels)
