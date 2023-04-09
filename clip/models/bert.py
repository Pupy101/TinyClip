from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import BertConfig
from transformers import BertModel as BaseBertModel

from clip.types import TextModel


class Bert(TextModel):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BaseBertModel(config=config)

    @property
    def output_shape(self) -> int:
        return self.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        output = self.bert.forward(input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

    @staticmethod
    def mean_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        weighted_embeddings = embeddings * attention_mask.unsqueeze(-1)
        mean_embeddings = torch.mean(weighted_embeddings, dim=1)
        return F.normalize(mean_embeddings, dim=1)


__all__ = ["Bert", "BertConfig"]
