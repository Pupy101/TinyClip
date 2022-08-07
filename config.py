from torch import nn, optim

from src.data import augmentations, transform_xlnet
from src.models import CLIP, ConvNeXt, TextPartCLIP, VisionPartCLIP, XLNet
from src.types import (
    ConvNeXtConfig,
    DataConfig,
    MultiTaskProportion,
    TrainConfig,
    XLNetConfig,
)
from src.utils.losses import FocalLoss

# data config
data_config = DataConfig(
    train_clip_csv="/content/train_clip.csv",
    valid_clip_csv="/content/valid_clip.csv",
    clip_name_image_column="image",
    clip_name_text_column="text",
    clip_batch_size_train=80,
    clip_batch_size_valid=128,
    train_image_classification_csv="/content/train_image.csv",
    valid_image_classification_csv="/content/valid_image.csv",
    image_classification_name_image_column="image",
    image_classification_name_label_column="label",
    image_classification_count_classes=1000,
    image_classification_batch_size_train=40,
    image_classification_batch_size_valid=64,
    train_masked_lm_csv="/content/train_text.csv",
    valid_masked_lm_csv="/content/valid_text.csv",
    masked_lm_name_text_column="text",
    masked_lm_batch_size_train=40,
    masked_lm_batch_size_valid=64,
    cls_token_ids=2,
    pad_token_ids=0,
    mask_token_idx=3,
    tokens_max_len=40,
    masked_portion=0.25,
    image_augmentations=augmentations,
    text_tokenizer_checkpoint="/content/bpe_8000",
    num_workers=4,
    mask_text_transform=transform_xlnet,
)

# Model parameters
clip_inner_shape = 256

# Vision part
count_classes = 1000  # count classes for classical ImageNet1k
vision_model_config = ConvNeXtConfig(
    in_channels=3,
    out_channels=clip_inner_shape,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)
vision_model = ConvNeXt(config=vision_model_config)
vision_part = VisionPartCLIP(
    model=vision_model, output_model_shape=clip_inner_shape, count_classes=count_classes
)

# Text part
text_model_config = XLNetConfig(
    num_layers=4,
    vocab_size=8_000,
    model_dim=256,
    num_heads=4,
    feedforward_dim=1024,
    dropout=0.5,
    activation="gelu",
    pad_idx=0,
)
text_model = XLNet(config=text_model_config)
text_part = TextPartCLIP(
    model=text_model,
    output_model_shape=text_model_config.model_dim,
    count_tokens=text_model_config.vocab_size,
    output_clip_shape=clip_inner_shape,
)

# CLIP
clip_model = CLIP(vision_part=vision_part, text_part=text_part)

# Optimizer
optimizer = optim.AdamW(clip_model.parameters(), lr=1e-2)

# scheduler
scheduler = None

# criterions
clip_criterion = FocalLoss()
criterion_image = nn.CrossEntropyLoss()
criterion_text = nn.CrossEntropyLoss(ignore_index=data_config.pad_token_ids)

# Coefficients for multitask learning
coefficients = MultiTaskProportion(clip=1, image=0.5, text=0.5)

# train config
train_config = TrainConfig(
    train_clip=True,
    train_image=False,
    train_text=False,
    n_epochs=5,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion_clip=clip_criterion,
    criterion_image=criterion_image,
    criterion_text=criterion_text,
    device="cuda",
    save_dir="./checkpoints",
    coefficients=coefficients,
    count_accumulated_batches=10,
    seed=1234,
)
