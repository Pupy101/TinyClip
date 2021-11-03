from typing import Tuple

import torch
import numpy as np

from torch import nn
from torchvision import models
from transformers import DistilBertForSequenceClassification


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
        self.matrix_normalize_img_emb = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=output_dim_img, out_features=overall_dim)
        )
        self.matrix_normalize_text_emb = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=output_dim_text, out_features=overall_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_img_net(self, img: torch.Tensor) -> torch.Tensor:
            img_embedding = self.model_img_emb(img)
            normalized_img_emb = self.matrix_normalize_img_emb(img_embedding)
            image_features = normalized_img_emb / normalized_img_emb.norm(dim=-1, keepdim=True)
            return image_features

    def forward_txt_net(self, txt: torch.Tensor) -> torch.Tensor:
            text_embedding = self.model_text_emb(txt)
            normalized_text_emb = self.matrix_normalize_text_emb(text_embedding)
            text_features = normalized_text_emb / normalized_text_emb.norm(dim=-1, keepdim=True)
            return text_features

    def forward_cosine_similarity(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor
        ) -> Tuple[torch.Tensor]:
        logit_scale = self.logit_scale.exp()
        logits_image = logit_scale * image_features @ text_features.t()
        logits_text = logits_image.t()
        return logits_image, logits_text

    def forward(
            self,
            vectors: Tuple[torch.Tensor],
            image_features: torch.Tensor = None,
            text_features: torch.Tensor = None,
            is_raw_output: bool = False
    ) -> Tuple[torch.Tensor]:
        img, text = vectors

        if image_features is None:
            image_features = self.forward_img_net(img)

        if image_features is None:
            text_features = self.forward_txt_net(text)
        
        logits_image, logits_text = self.forward_cosine_similarity(image_features, text_features)

        if is_raw_output:
            logits_image, logits_text, image_features, text_features

        return logits_image, logits_text

    @torch.no_grad()
    def inference(
            self,
            input_tensor: Tuple[torch.Tensor],
            is_rewrite_classes: bool = False
    ) -> torch.Tensor:
        img, text_classes = input_tensor
        if hasattr(self, 'inference_text_embedding') and not is_rewrite_classes:
            inference_text_embedding = self.inference_text_embedding
            logits_image, logits_text = self.forward(img, text_classes, text_features=inference_text_embedding)
        else:
            logits_image, logits_text, image_features, text_features = self.forward(img, text_classes, is_raw_output=True)
            self.inference_text_embedding = text_features
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
