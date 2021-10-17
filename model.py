from torchvision import models
from torch import nn


class TextEmbedding(nn.Module):

    def __init__(
            self,
            tokenizer,
            num_embeddings,
            embedding_dim,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            bidirectional
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def forward(self, text):
        tokenized_text = self.tokenizer(text)
        embedding_text = self.embedding(tokenized_text)
        hidden = self.lstm(embedding_text)

class ImageEmbedding(nn.Module):

    def __init__(self, model):
        self.model = model
    
    def forward(self, img):
        hidden = self.model(img)
        return hidden
