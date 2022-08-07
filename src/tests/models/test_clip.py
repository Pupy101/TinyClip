import pytest
import torch

from src.models import CLIP, ConvNeXt, TextPartCLIP, VisionPartCLIP, XLNet
from src.types import ConvNeXtConfig, XLNetConfig

default_clip_shape = 256
default_count_classes = 1000
default_convnext = ConvNeXtConfig(
    in_channels=3,
    out_channels=default_clip_shape,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)
default_xlnet = XLNetConfig(
    num_layers=4,
    vocab_size=8_000,
    model_dim=256,
    num_heads=4,
    feedforward_dim=1024,
    dropout=0.5,
    activation="gelu",
    pad_idx=0,
)

custom_clip_shape = 512
custom_count_classes = 50
custom_convnext = ConvNeXtConfig(
    in_channels=3,
    out_channels=custom_clip_shape,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)
custom_xlnet = XLNetConfig(
    num_layers=4,
    vocab_size=8_000,
    model_dim=768,
    num_heads=12,
    feedforward_dim=3072,
    dropout=0.5,
    activation="gelu",
    pad_idx=0,
)


@pytest.mark.parametrize(
    ["convnext_config", "xlnet_config", "clip_shape", "count_classes"],
    [
        (default_convnext, default_xlnet, default_clip_shape, default_count_classes),
        (custom_convnext, custom_xlnet, custom_clip_shape, custom_count_classes),
    ],
)
def test_clip(convnext_config, xlnet_config, clip_shape, count_classes):
    vision_model = VisionPartCLIP(
        model=ConvNeXt(config=convnext_config),
        output_model_shape=clip_shape,
        count_classes=count_classes,
    )
    text_model = TextPartCLIP(
        model=XLNet(config=xlnet_config),
        output_model_shape=xlnet_config.model_dim,
        count_tokens=default_xlnet.vocab_size,
        output_clip_shape=clip_shape,
    )
    clip = CLIP(vision_model, text_model)
    image = torch.rand(4, 3, 224, 224)
    text = torch.arange(2048, dtype=torch.long).reshape(4, -1)
    with torch.no_grad():
        output = clip.forward(image, text)
    assert tuple(output.logits.image.shape) == (4, 4)
    assert tuple(output.logits.text.shape) == (4, 4)
    assert tuple(output.embeddings.image.shape) == (4, clip_shape)
    assert tuple(output.embeddings.text.shape) == (4, clip_shape)
    with torch.no_grad():
        output = clip.vision_part.forward(image, is_classification=True)
    assert tuple(output.shape) == (4, count_classes)
    with torch.no_grad():
        output = clip.text_part.forward(text, is_mlm=True)
    assert tuple(output.shape) == (4, xlnet_config.vocab_size, text.size(1))
