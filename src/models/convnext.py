"""
Model adapted from
https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py
"""

import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.types import ConvNeXtConfig


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, input_: Tensor) -> Tensor:
        x = input_
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class ConvBlock(nn.Module):
    def __init__(self, dim: int, drop_proba: float = 0.0) -> None:
        super().__init__()
        self.drop_proba = drop_proba
        self.dwconv = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = LayerNorm(normalized_shape=dim)
        self.pwconv1 = nn.Linear(in_features=dim, out_features=4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(in_features=4 * dim, out_features=dim)

    def forward(self, input_: Tensor) -> Tensor:
        if self.training:
            if random.random() < self.drop_proba:
                return input_
        x = self.dwconv(input_)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input_ + x.permute(0, 3, 1, 2)


class DownSampleBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(
            normalized_shape=in_channels, data_format="channels_first"
        )
        self.down_sample_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self.down_sample_conv(self.norm(input_))


class ConvNeXt(nn.Module):
    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__()
        self.config = config
        assert config.depths is not None
        assert config.dims is not None
        assert len(config.depths) == len(config.dims)
        self.count_stages = len(config.dims)
        self.downsample_stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        config.in_channels, config.dims[0], kernel_size=4, stride=4
                    ),
                    LayerNorm(
                        normalized_shape=config.dims[0], data_format="channels_first"
                    ),
                )
            ]
        )
        self.downsample_stages.extend(
            [
                DownSampleBlock(
                    in_channels=config.dims[i], out_channels=config.dims[i + 1]
                )
                for i in range(len(config.dims) - 1)
            ]
        )
        self.convnext_stages = nn.ModuleList([])
        dp_rates = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]
        cur_ind = 0
        for i in range(len(config.dims)):
            self.convnext_stages.append(
                nn.Sequential(
                    *[
                        ConvBlock(dim=config.dims[i], drop_proba=dp_rates[cur_ind + j])
                        for j in range(config.depths[i])
                    ]
                )
            )
            cur_ind += config.depths[i]
        self.last_norm = LayerNorm(config.dims[-1])
        self.head = nn.Linear(config.dims[-1], config.out_channels)

    def forward_features(self, input_: Tensor) -> Tensor:
        x = input_
        for i in range(self.count_stages):
            x = self.downsample_stages[i](x)
            x = self.convnext_stages[i](x)
        return self.last_norm(x.mean([-2, -1]))

    def forward(self, input_: Tensor) -> Tensor:
        x = self.forward_features(input_)
        x = self.head(x)
        return x
