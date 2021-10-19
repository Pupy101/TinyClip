from typing import Callable, Union, Tuple

import torch

from torch import nn
from torchvision import models
from transformers import DistilBertForSequenceClassification


class CosineSimilarity2DVectors(nn.Module):

    def __init__(
            self,
            eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps

    def forward(
            self,
            vectors: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        first_vector, second_vector = vectors
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
        super().__init__()
        if overall_dim is None:
            overall_dim = max(output_dim_img, output_dim_text)
        self.model_img_emb = image_embedding
        self.clf_img = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=output_dim_img, out_features=overall_dim)
        )
        self.model_text_emb = text_embedding
        self.clf_text = nn.Sequential(
            text_embedding,
            nn.ReLU(),
            nn.Linear(in_features=output_dim_text, out_features=overall_dim)
        )

        self.cosine_similarity = CosineSimilarity2DVectors()

    def forward(
            self,
            vectors: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        img, text = vectors
        img_emb = self.model_img_emb(img)
        text_emb = self.model_text_emb(text)
        classes_img = self.clf_img(img_emb)
        classes_text = self.clf_text(text_emb)
        output = self.cosine_simularity((classes_img, classes_text))
        return output

    def inference(
            self,
            input_tensor: Tuple[torch.Tensor],
            is_rewrite_classes: bool = False
    ) -> torch.Tensor:
        img, text_classes = input_tensor
        if hasattr(self, 'classes') and not is_rewrite_classes:
            classes_text = self.classes
        else:
            text_emb = self.model_text_emb(text_classes)
            classes_text = self.clf_text(text_emb)
            self.classes = classes_text
        img_emb = self.model_img_emb(img)
        classes_img = self.clf_img(img_emb)
        output = self.cosine_simularity((classes_img, classes_text))
        return torch.argmax(output, dim=1)


mobilenet_v3 = models.mobilenet_v3_small(pretrained=True)
mobilenet_v3.classifier = nn.Identity()
img_dim_size = 576


class DistilBertForCLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.bert.classifier = nn.Identity()

    def forward(self, x):
        return self.bert(x)['logits']


distilbert = DistilBertForCLIP()
text_dim_size = 768

clip = CLIP(
    image_embedding=mobilenet_v3,
    text_embedding=distilbert,
    output_dim_img=img_dim_size,
    output_dim_text=text_dim_size,
    overall_dim=512
)
