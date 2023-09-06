import torch.nn as nn
import torch


class TimeSeriesBERTEmbedding(nn.Module):
    def __init__(self, embedding_size, time_series_max_len):
        super().__init__()
        self.embedding_size = embedding_size

        self.value_embedding = nn.Linear(1, embedding_size)

        self.masked_value_embedding = nn.Parameter(torch.empty(embedding_size), requires_grad=True)
        nn.init.uniform_(self.masked_value_embedding)

        self.pos_embedding = TimeSeriesBERTPositionalEmbedding(embedding_size, time_series_max_len)

    def forward(self, x):
        pos_matrix = torch.arange(x.size(1)).to(x.device)
        pos_matrix = pos_matrix.unsqueeze(0).unsqueeze(0)

        mask = (x == -10).to(x.device)
        x = self.value_embedding(x)
        x = x.where(mask == False, self.masked_value_embedding)

        return x + self.pos_embedding(pos_matrix)

    def init_weights(self, distribution_module):
        distribution_module(self.value_embedding.weight)


class TimeSeriesBERTPositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, time_series_max_len):
        super().__init__()

        self.pe = nn.Parameter(torch.empty(time_series_max_len, embedding_size))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, pos_matrix):
        return self.pe[pos_matrix]
