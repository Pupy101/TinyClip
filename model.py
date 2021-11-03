from typing import Tuple

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
            normalize: bool = False
    ) -> torch.Tensor:
        first_vector, second_vector = vectors
        assert len(first_vector.shape) == len(second_vector.shape) == 2, 'Vectors dimesions must be 2'
        first_l2_norm = torch.linalg.norm(first_vector, dim=1).unsqueeze(1)
        second_l2_norm = torch.linalg.norm(second_vector, dim=1).unsqueeze(0)
        multiply_of_vectors = torch.sum(
            first_vector.unsqueeze(2) * second_vector.permute(1, 0).unsqueeze(0),
            dim=1
        )
        if normalize:
            return multiply_of_vectors / (torch.sqrt(first_l2_norm * second_l2_norm) + self.eps)
        return multiply_of_vectors


class CLIP(nn.Module):

    def __init__(
            self,
            name_image_embedding: str,
            name_text_embedding: str,
            overall_dim: int = None
    ):
        """
        name_image_embedding - name model for embedding image
        name_text_embedding - name model for embedding text
        overall_dim - overall dimesion
        """
        super().__init__()
        self.model_img_emb, output_dim_img = configuration_image_model(name_image_embedding)
        self.model_text_emb, output_dim_text = configuration_text_model(name_text_embedding)
        if overall_dim is None:
            overall_dim = max(output_dim_img, output_dim_text)
        self.clf_img = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=output_dim_img, out_features=overall_dim)
        )
        self.clf_text = nn.Sequential(
            nn.LeakyReLU(),
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
        output = self.cosine_similarity((classes_img, classes_text))
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
        output = self.cosine_similarity((classes_img, classes_text), normalize=True)
        return torch.argmax(output, dim=1)

    @property
    def device(self):
        return next(iter(self.parameters())).device


def configuration_image_model(name_model: str) -> Tuple[nn.Module, int]:
    if name_model == 'mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier = nn.Identity()
        img_dim_size = 576
    elif name_model == 'wide_resnet50':
        model = models.wide_resnet50_2(pretrained=True)
        model.fc = nn.Identity()
        img_dim_size = 2048
    elif name_model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Identity()
        img_dim_size = 2048        
    return model, img_dim_size


class WrapperModelFromHuggingFace(nn.Module):

    def __init__(self, hugging_face_model: nn.Module):
        super().__init__()
        self.model = hugging_face_model
        

    def forward(self, x) -> torch.Tensor:
        return self.model(x)['logits']


def configuration_text_model(name_model: str) -> Tuple[nn.Module, int]:
    if name_model == 'distilbert':
        bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        bert.classifier = nn.Identity()
        model = WrapperModelFromHuggingFace(bert)
        text_dim_size = 768    
    return model, text_dim_size
