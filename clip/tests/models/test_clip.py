import pytest
import torch

from clip.models import (
    CLIP,
    Bert,
    BertConfig,
    ConvNeXt,
    ConvNeXtConfig,
    ImagePartCLIP,
    TextPartCLIP,
)

default_clip_shape = 256
default_count_classes = 1000
default_convnext = ConvNeXtConfig(
    in_channels=3,
    out_channels=default_clip_shape,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)
default_deberta = BertConfig(
    num_hidden_layers=4,
    vocab_size=8_000,
    hidden_size=default_clip_shape,
    num_attention_heads=4,
    intermediate_size=1024,
    position_embedding_type="relative_key_query",
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.5,
    hidden_act="gelu",
    max_pos_emb=512,
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
custom_deberta = BertConfig(
    num_hidden_layers=4,
    vocab_size=8_000,
    hidden_size=custom_clip_shape,
    num_attention_heads=4,
    intermediate_size=1024,
    position_embedding_type="relative_key_query",
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.5,
    hidden_act="gelu",
    max_pos_emb=512,
)


@pytest.mark.parametrize(
    ["convnext_config", "deberta_config", "count_classes"],
    [
        (default_convnext, default_deberta, default_count_classes),
        (custom_convnext, custom_deberta, custom_count_classes),
    ],
)
def test_clip(
    convnext_config: ConvNeXtConfig,
    deberta_config: BertConfig,
    count_classes: int,
    device: torch.device,
):
    vision_model = ImagePartCLIP(
        model=ConvNeXt(config=convnext_config),
        count_classes=count_classes,
    )
    text_model = TextPartCLIP(model=Bert(config=deberta_config))
    clip = CLIP(vision_model, text_model).to(device)
    image = torch.rand(4, 3, 224, 224).to(device)
    text = torch.arange(2048, dtype=torch.long).reshape(4, -1).to(device)
    with torch.no_grad():
        output = clip.forward(image, text)
    assert tuple(output.logits.image.shape) == (4, 4)
    assert tuple(output.logits.text.shape) == (4, 4)
    assert tuple(output.embeddings.image.shape) == tuple(output.embeddings.text.shape)
    with torch.no_grad():
        output_image = clip.image_part.forward(image, classification=True)
    assert tuple(output_image.shape) == (4, count_classes)
    with torch.no_grad():
        output_text = clip.text_part.forward(text, masked_lm=True)
    assert tuple(output_text.shape) == (4, deberta_config.vocab_size, text.size(1))
