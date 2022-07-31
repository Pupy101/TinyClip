import torch

from src.models import CLIP, ConvNeXt, TextPartCLIP, VisionPartCLIP, XLNet
from src.types import ConvNeXtConfig, XLNetConfig

default_convnext = ConvNeXtConfig(
    in_channels=3,
    out_channels=256,
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
output_shape = 256


def test_clip():
    vision_model = VisionPartCLIP(
        ConvNeXt(config=default_convnext),
        output_model_shape=output_shape,
        count_classes=1000,
    )
    text_model = TextPartCLIP(
        XLNet(config=default_xlnet),
        output_model_shape=output_shape,
        count_tokens=default_xlnet.vocab_size,
        output_shape=output_shape,
    )
    clip = CLIP(vision_model, text_model)
    image = torch.rand(4, 3, 224, 224)
    text = torch.arange(2048, dtype=torch.long).reshape(4, -1)
    with torch.no_grad():
        output = clip.forward(image, text)
    assert tuple(output.logits.image.shape) == (4, 4)
    assert tuple(output.logits.text.shape) == (4, 4)
    assert tuple(output.embeddings.image.shape) == (4, 256)
    assert tuple(output.embeddings.text.shape) == (4, 256)
