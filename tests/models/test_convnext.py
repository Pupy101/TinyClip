from itertools import product
from typing import Any, Dict, List, Union

import pytest
import torch
from transformers import ConvNextConfig, ConvNextModel, ConvNextV2Config, ConvNextV2Model

from clip.models import create_convnext, create_convnext_v2
from clip.types import ImageModelType


@pytest.mark.parametrize(
    ["model_type", "depths", "hidden_sizes", "drop_path_rate", "hidden_act"],
    product(
        [ImageModelType.CONVNEXT.value, ImageModelType.CONVNEXTV2.value],
        [[3, 3, 9, 3], [3, 3, 27, 3]],
        [[96, 192, 384, 768], [128, 256, 512, 1024]],
        [0.5],
        ["gelu", "gelu_new"],
    ),
)
def test_custom_convnext(
    model_type: str,
    depths: List[int],
    hidden_sizes: List[int],
    drop_path_rate: float,
    hidden_act: str,
    device: torch.device,
) -> None:
    batch_size = 2
    kwargs: Dict[str, Any] = {
        "drop_path_rate": drop_path_rate,
        "depths": depths,
        "hidden_sizes": hidden_sizes,
        "hidden_act": hidden_act,
    }
    config: Union[ConvNextConfig, ConvNextV2Config]
    model: Union[ConvNextModel, ConvNextV2Model]
    if model_type == ImageModelType.CONVNEXT.value:
        config, model = create_convnext(**kwargs)
    elif model_type == ImageModelType.CONVNEXTV2.value:
        config, model = create_convnext_v2(**kwargs)
    else:
        raise RuntimeError
    _ = model.to(device)
    input_tensor = torch.rand(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model.forward(input_tensor).pooler_output
    assert tuple(output.shape) == (batch_size, config.hidden_sizes[-1])
