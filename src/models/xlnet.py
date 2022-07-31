from typing import Optional

import torch
from torch import Tensor, nn
from transformers import XLNetConfig as BaseXLNetConfig
from transformers import XLNetModel

from src.types import XLNetConfig


class XLNet(nn.Module):
    def __init__(self, config: XLNetConfig) -> None:
        super().__init__()
        self.config = config
        xlnet_config = BaseXLNetConfig(
            vocab_size=config.vocab_size,
            d_model=config.model_dim,
            n_layer=config.num_layers,
            n_head=config.num_heads,
            d_inner=config.feedforward_dim,
            ff_activation=config.activation,
            dropout=config.dropout,
        )
        self.xlnet = XLNetModel(config=xlnet_config)
        self.pad_idx = config.pad_idx

    def forward_xlnet(
        self,
        input_ids: Tensor,
        perm_mask: Optional[Tensor] = None,
        target_mapping: Optional[Tensor] = None,
    ) -> Tensor:
        attn_mask = self.create_attn_mask(input_ids)
        output = self.xlnet.forward(
            input_ids,
            attention_mask=attn_mask,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
        )
        return output.last_hidden_state

    def forward(
        self,
        input_ids: Tensor,
        perm_mask: Optional[Tensor] = None,
        target_mapping: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward_xlnet(
            input_ids, perm_mask=perm_mask, target_mapping=target_mapping
        )

    def create_attn_mask(self, input_ids: Tensor) -> Tensor:
        mask = input_ids != self.pad_idx
        return mask.to(torch.float)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
