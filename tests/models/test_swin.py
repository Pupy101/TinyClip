from itertools import product
from typing import Any, Dict, List, Union

import pytest
import torch
from transformers import SwinConfig, SwinModel, Swinv2Config, Swinv2Model

from clip.models import create_swin, create_swin_v2
from clip.types import ImageModelType


@pytest.mark.parametrize(
    ["model_type", "depths", "num_heads", "embed_dim", "drop_path_rate", "hidden_act"],
    product(
        [ImageModelType.SWIN.value, ImageModelType.SWINV2.value],
        [[2, 2, 2, 2], [2, 2, 4, 2]],
        [[3, 4, 6, 8], [3, 4, 8, 12]],
        [48, 96],
        [0.5],
        ["gelu", "gelu_new"],
    ),
)
def test_custom_swin(
    model_type: str,
    depths: List[int],
    num_heads: List[int],
    embed_dim: int,
    drop_path_rate: float,
    hidden_act: str,
) -> None:
    batch_size = 2
    kwargs: Dict[str, Any] = {
        "depths": depths,
        "num_heads": num_heads,
        "embed_dim": embed_dim,
        "drop_path_rate": drop_path_rate,
        "hidden_act": hidden_act,
    }
    config: Union[SwinConfig, Swinv2Config]
    model: Union[SwinModel, Swinv2Model]
    if model_type == ImageModelType.SWIN.value:
        config, model = create_swin(**kwargs)
    elif model_type == ImageModelType.SWINV2.value:
        config, model = create_swin_v2(**kwargs)
    else:
        raise RuntimeError
    _ = model.to("cpu")
    input_tensor = torch.rand(batch_size, 3, 224, 224).to("cpu")
    with torch.no_grad():
        output = model.forward(input_tensor).pooler_output
    assert tuple(output.shape) == (batch_size, config.embed_dim * 8)
