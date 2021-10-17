from typing import Callable, Union

import torch

from torch import nn
from torchvision import models
from transformers import DistilBertTokenizer, DistilBertForMaskedLM


class EmbeddingModel(nn.Module):

    def __init__(
            self,
            model: nn.Module
    ):
        """
        model - torch model with embedding at output
        """
        super().__init__()
        self.model = model
    
    def forward(
        self,
        input_tensor: Union[torch.Tensor, ]
    ) -> torch.Tensor:
        hidden = self.model(input_tensor)
        return hidden


class CosineSimilarity2DVectors(nn.Module):
    
    def __init__(
        self,
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        first_vector: torch.Tensor,
        second_vector: torch.Tensor
    ) -> torch.Tensor:
        assert len(first_vector.shape) == len(second_vector.shape) == 2, 'Vectors dimesions must be 2'
        first_l2_norm = torch.linalg.norm(first_vector, dim=1).unsqueeze(1)
        second_l2_norm = torch.linalg.norm(second_vector, dim=1).unsqueeze(0)
        multiply_of_vectors = torch.sum(
            first_vector.unsqueeze(2) * second_vector.permute(1, 0).unsqueeze(0),
            dim=1
        )
        return multiply_of_vectors / (torch.sqrt(first_l2_norm * second_l2_norm) + self.eps)


class CLIP(nn.Module):

    def __init__(
        self,
        image_embedding: nn.Module,
        text_embedding: nn.Module,
        output_dim_img: int,
        output_dim_text: int,
        overall_dim: int = None
    ):
        """
        image_embedding - model for embedding image
        text_embedding - model for embedding text
        output_dim_img - dim of output image_embedding
        output_dim_text - dim of output text_embedding
        overall_dim - overall dimesion
        """
        if overall_dim is None:
            overall_dim = max(output_dim_img, output_dim_text)

        self.model_img_emb = nn.Sequential(
            image_embedding,
            nn.ReLU(),
            nn.Linear(output_dim_img, overall_dim)
        )
        
        self.model_text_emb = nn.Sequential(
            text_embedding,
            nn.ReLU(),
            nn.Linear(output_dim_text, overall_dim)
        )

        self.cosine_simularity = nn.CosineSimilarity2DVectors()
    
    def forward(
        self,
        img: torch.Tensor,
        text: torch.Tensor
    ) -> torch.Tensor:
        img_emb = self.model_img_emb(img)
        text_emb = self.model_text_emb(text)
        output = self.cosine_simularity(img_emb, text_emb)
        return output
    
    def inference(
        self,
        img: torch.Tensor,
        text_classes: torch.Tensor,
        is_rewrite_classes: bool
    ) -> torch.Tensor:
        if hasattr(self, 'classes') and not is_rewrite_classes:
            classes = self.classes
        else:
            classes = self.model_text_emb(text_classes)
            self.classes = classes
        img_emb = self.model_img_emb(img)
        output = self.cosine_simularity(img_emb, classes)
        return torch.argmax(output, dim=1)
        
