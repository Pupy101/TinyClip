"""
Model adapted from
https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py
"""

import random

import torch
from blocks.layer import DWConv2d, LayerNorm2d
from torch import Tensor, nn

from clip.types import ConvNeXtConfig, ImageModel


class ConvBlock(nn.Module):
    def __init__(self, dim: int, drop_proba: float = 0.0) -> None:
        super().__init__()
        self.drop_proba = drop_proba
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = LayerNorm2d(channels=dim)
        self.pwconv1 = nn.Linear(in_features=dim, out_features=4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(in_features=4 * dim, out_features=dim)

    def forward(self, batch: Tensor) -> Tensor:
        if self.training and random.random() < self.drop_proba:
            return batch
        in_batch = batch
        batch = self.conv(batch)
        batch = self.norm(batch)
        batch = self.pwconv1(batch.permute(0, 2, 3, 1))
        batch = self.act(batch)
        batch = self.pwconv2(batch)
        return in_batch + batch.permute(0, 3, 1, 2)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dw_conv: bool = False) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels=in_channels)
        conv = DWConv2d if use_dw_conv else nn.Conv2d
        self.down_conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, batch: Tensor) -> Tensor:
        return self.down_conv(self.norm(batch))


class ConvNeXt(ImageModel):
    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__()
        self.config = config
        conv = DWConv2d if config.use_dw_conv else nn.Conv2d
        self.count_stages = len(config.dims)
        self.downsample_stages = nn.ModuleList(
            [
                nn.Sequential(
                    conv(config.in_channels, config.dims[0], kernel_size=4, stride=4),
                    LayerNorm2d(config.dims[0]),
                )
            ]
        )
        self.downsample_stages.extend(
            [
                DownSampleBlock(
                    in_channels=config.dims[i],
                    out_channels=config.dims[i + 1],
                    use_dw_conv=config.use_dw_conv,
                )
                for i in range(self.count_stages - 1)
            ]
        )
        self.conv_stages = nn.ModuleList([])
        dp_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        cur_ind = 0
        for i in range(self.count_stages):
            self.conv_stages.append(
                nn.Sequential(
                    *[
                        ConvBlock(dim=config.dims[i], drop_proba=dp_rates[cur_ind + j])
                        for j in range(config.depths[i])
                    ]
                )
            )
            cur_ind += config.depths[i]
        self.last_norm = nn.LayerNorm(normalized_shape=config.dims[-1])
        self.head = nn.Linear(config.dims[-1], config.out_channels)

    @property
    def output_shape(self) -> int:
        return self.config.out_channels

    def forward(self, batch: Tensor) -> Tensor:
        for downsample, conv in zip(self.downsample_stages, self.conv_stages):
            batch = conv(downsample(batch))
        batch = self.last_norm(batch.mean([-2, -1]))
        return self.head(batch)


__all__ = ["ConvNeXt", "ConvNeXtConfig"]
