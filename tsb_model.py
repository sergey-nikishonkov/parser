from collections import OrderedDict

from tsb_embedding import TimeSeriesBERTEmbedding

import math

import torch
from torch import nn
from torch.nn import functional as F


class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    def init_weights(self, distribution_module):
        pass


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_prob):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNormalization(size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, sublayer):
        x = x + sublayer(x)
        return self.dropout(self.layer_norm(x))


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ff_hidden_size, dropout_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, ff_hidden_size)
        self.w_2 = nn.Linear(ff_hidden_size, d_model)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

    def init_weights(self, distribution_module):
        distribution_module(self.w_1.weight)
        distribution_module(self.w_2.weight)


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout_layer=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores /= math.sqrt(query.size(-1))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probabilites = F.softmax(attention_scores, dim=-1)

        if dropout_layer is not None:
            attention_probabilites = dropout_layer(attention_scores)

        return torch.matmul(attention_probabilites, value), attention_probabilites


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads_count, dropout_prob=0.1):
        super().__init__()

        # d_key = d_query
        d_key = d_model // heads_count
        self.d_key = d_key

        self.heads_count = heads_count

        # Weight Matrixes for Q, K, V
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_key * heads_count) for _ in range(3)])
        self.output_linear_layer = nn.Linear(d_model, d_model)

        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            linear_layer(x).view(batch_size, -1, self.heads_count, self.d_key).transpose(1, 2)
            for linear_layer, x in zip(self.linear_layers, (query, key, value))
        ]

        x, attention_probabilites = self.attention(
            query=query, key=key, value=value, mask=mask, dropout_layer=self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads_count * self.d_key)

        return self.output_linear_layer(x)

    def init_weights(self, distribution_module):
        distribution_module(self.linear_layers[0].weight)
        distribution_module(self.linear_layers[1].weight)
        distribution_module(self.linear_layers[2].weight)
        distribution_module(self.output_linear_layer.weight)


class EncoderLayer(nn.Module):
    def __init__(self, heads_count, hidden_size, ff_hidden_size, dropout_prob=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=hidden_size, heads_count=heads_count)
        self.input_sublayer = SublayerConnection(size=hidden_size, dropout_prob=dropout_prob)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, ff_hidden_size=ff_hidden_size, dropout_prob=dropout_prob
        )
        self.output_sublayer = SublayerConnection(size=hidden_size, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.multi_head_attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

    def init_weights(self, distribution_module):
        self.multi_head_attention.init_weights(distribution_module)
        self.feed_forward.init_weights(distribution_module)


class TimeSeriesBERTModel(nn.Module):
    def __init__(self, time_series_size=71, hidden_size=128, encoder_layers_count=4, heads_count=16, dropout_prob=0.1):
        """
        Time Series BERT Model
        Parameters
        ----------
        :param time_series_size: size of input time series(Int)
        :param hidden_size: size of hidden layers(Int)
        :param encoder_layers_count: number of encoder layers(Int)
        :param heads_count: number of heads in encoder layer(Int)
        :param dropout_prob: probability of dropout layer(Float)
        """

        super().__init__()

        self.time_series_size = time_series_size
        self.hidden_size = hidden_size
        self.ff_hidden_size = hidden_size * 2
        self.heads_count = heads_count
        self.dropout_prob = dropout_prob

        self.tsb_embedding = TimeSeriesBERTEmbedding(
            embedding_size=self.hidden_size, time_series_max_len=time_series_size
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=self.hidden_size,
                    ff_hidden_size=self.ff_hidden_size,
                    heads_count=self.heads_count,
                    dropout_prob=self.dropout_prob,
                )
                for _ in range(encoder_layers_count)
            ]
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.tsb_embedding(x.reshape(batch_size, self.time_series_size, -1)).reshape(
            batch_size, self.time_series_size, -1
        )

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x

    def init_weights(self, distribution_module):
        self.tsb_embedding.init_weights(distribution_module)
        for encoder_layer_index in range(len(self.encoder_layers)):
            self.encoder_layers[encoder_layer_index].init_weights(distribution_module)

    def set_weights(self, weights):  # TODO: Specify type of weights and return type
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def get_weights(self) -> NDArrays:  # TODO: Specify return type
        return [val.cpu().numpy() for _, val in self.state_dict().items()]


class TimeSeriesBERTModelForTraining(nn.Module):
    def __init__(self, tsb_model):
        super().__init__()
        self.tsb_model = tsb_model
        self.output_layer = nn.Linear(
            self.tsb_model.hidden_size * self.tsb_model.time_series_size, self.tsb_model.time_series_size
        )

    def forward(self, x):
        batch_size = x.size(0)
        time_series_size = x.size(1)

        model_output = self.tsb_model(x).reshape(
            batch_size, self.tsb_model.hidden_size * self.tsb_model.time_series_size
        )

        return self.output_layer(model_output).reshape(batch_size, time_series_size, -1)

    def init_weights(self, distribution_module):
        distribution_module(self.output_layer.weight)
        self.tsb_model.init_weights(distribution_module)

    def set_weights(self, weights: NDArrays):  # TODO: Specify type of weights and return type
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def get_weights(self) -> NDArrays:  # TODO: Specify return type
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
