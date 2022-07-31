from torch import nn, optim

from src.data import augmentations, transform_xlnet
from src.types import (
    ConvNeXtConfig,
    DataConfig,
    MultiTaskProportion,
    TrainConfig,
    XLNetConfig,
)
from src.utils.losses import FocalLoss

image_model_config = ConvNeXtConfig(
    in_channels=3,
    out_channels=256,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
)
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

clip_shape = 256

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
    num_workers=2,
    mask_text_transform=transform_xlnet,
)

train_config = TrainConfig(
    n_epochs=5,
    optimizer=optim.AdamW,
    criterion_clip=FocalLoss(),
    criterion_image=nn.CrossEntropyLoss(),
    criterion_text=nn.CrossEntropyLoss(ignore_index=data_config.pad_token_ids),
    device="cuda",
    save_dir="./checkpoints",
    coefficients=MultiTaskProportion(),
    accumulation_steps=2,
    seed=1234,
)
