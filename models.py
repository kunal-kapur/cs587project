import torch
import numpy as np
from typing import List, cast
import torch.nn as nn
import math
import sys

class CrossNet(nn.Module):
    R"""
    Cross Net layer
    """
    def __init__(self, input_dim):
        super().__init__()

        self.alphas = nn.Parameter(torch.Tensor(input_dim, 1))
        self.bias = torch.nn.Parameter(torch.Tensor(input_dim))

        # Kaiming initialization
        # stdv = 1.0/math.sqrt(input_dim)
        # self.alphas.data.uniform_(-stdv, stdv)
        # self.alphas = self.alphas
        torch.nn.init.kaiming_uniform_(self.alphas,
                               a=0, mode="fan_in",
                               nonlinearity="relu")

        self.bias.data.zero_()

    def to(self, device):
        self.alphas = torch.nn.Parameter(self.alphas.to(device))
        self.bias = torch.nn.Parameter(self.bias.to(device))
        return super().to(device)

    def forward(self, input, X):
        # Input shape: torch.Size([minibatch, input_dim])

        bias = self.bias + X
        X = X.unsqueeze(2)
        input = input.unsqueeze(2)

        transposed_X = torch.transpose(X, 1, 2) # keep batch the same
        X = torch.bmm(X, transposed_X)

        val = torch.matmul(X, self.alphas).squeeze(2)

        return val + bias

class CrossNetV2(nn.Module):
    R"""
    Cross Net layer
    """
    def __init__(self, input_dim):
        super().__init__()

        self.alphas = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(input_dim))

        # Kaiming initialization
        stdv = 1.0/math.sqrt(input_dim)
        self.alphas.data.uniform_(-stdv, stdv)
        self.alphas = self.alphas
        self.bias.data.zero_()

    def to(self, device):
        self.alphas = torch.nn.Parameter(self.alphas.to(device))
        self.bias = torch.nn.Parameter(self.bias.to(device))
        return super().to(device)

    def forward(self, input, X):
        # Input shape: torch.Size([minibatch, input_dim])

        # X = X.unsqueeze(2)
        # input = input.unsqueeze(2)

        return input * (torch.matmul(X, self.alphas.t()) + self.bias) + X

        #return input.unsqueeze * ((torch.matmul(self.alphas, X.unsqueeze(-1)) + self.bias.unsqueeze).squeeze(-1)) + X


class DCN(nn.Module):
    R"""
    input_size: int, size of the input
    dcn_layer_len: number of cross layers we have
    layer_sizes: List[int], size of the hidden layers in the hidden network
    output_dim: int, size of the output layer
    embedding_dim: dimension size of the embedding, typically the size of the input dim
    """
    def __init__(self, categorical_features, num_numerical_features, dcn_layer_len, layer_sizes, concat_layer_sizes, output_dim, cross_net_V2=False, stacked=False):


        super().__init__()
        self.stacked = stacked
        self.embedding_layers = None
        num_embeddings = 0

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, round(6 * (num_categories) ** (1/4))) for num_categories in categorical_features
        ])
        num_embeddings = sum(round(6 * (num_categories) ** (1/4)) for num_categories in categorical_features)

        # final input dimension
        input_dim = num_embeddings + num_numerical_features

        self.cross_layers = None

        if not cross_net_V2:
            self.cross_layers = [CrossNet(input_dim=input_dim) for i in range(dcn_layer_len)]
        else:
            self.cross_layers = [CrossNetV2(input_dim=input_dim) for i in range(dcn_layer_len)]

        prev_dim = input_dim
        mlp = []
        for i in range(0, len(layer_sizes)):
            mlp.append(torch.nn.Linear(in_features=prev_dim, out_features=layer_sizes[i]))
            mlp.append(torch.nn.ReLU())
            mlp.append(nn.BatchNorm1d(layer_sizes[i]))
            prev_dim = layer_sizes[i]


        #mlp.pop() # Pop last ReLU layer?
        mlp.pop()
        mlp.pop()
        # mlp.append(nn.BatchNorm1d(layer_sizes[-1]))
        self.mlp = torch.nn.Sequential(*mlp)

        # the cross layers have input dim and we add to last layer size
        mlp_out_dim = input_dim
        cross_output_dim = 0
        if len(layer_sizes) > 0:
            mlp_out_dim = layer_sizes[-1]
        if dcn_layer_len > 0:
            cross_output_dim = input_dim # cross output is always input size
        final_layer_dim = input_dim + layer_sizes[-1] if not stacked else layer_sizes[-1]

        prev_size = mlp_out_dim + cross_output_dim
        self.concat_layer = []

        self.concat_layer.append(nn.BatchNorm1d(final_layer_dim))
        self.concat_layer.append(nn.Linear(in_features=final_layer_dim, out_features=output_dim))
        self.concat_layer.append(nn.Sigmoid()) # get logits
        self.concat_layer = torch.nn.Sequential(*self.concat_layer)

    def to(self, device):
        for i in range(len(self.embedding_layers)):
            self.embedding_layers[i] = self.embedding_layers[i].to(device)

        for i in range(len(self.cross_layers)):
            self.cross_layers[i] = self.cross_layers[i].to(device)
        return super().to(device)


    def forward(self, categorical_input, numerical_input):

        embedded_categorical = [
            embedding_layer(categorical_input[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)
        ]

        embedded_categorical = torch.cat(embedded_categorical, dim=1)  # Concatenate along feature dimension

        combined_input = torch.cat([embedded_categorical, numerical_input], dim=1)
        cross_output = combined_input

        if self.stacked:
            # stacked arch
            X = self.mlp(cross_output)
        else:
            # parallel arch
            mlp_output = self.mlp(combined_input)

        for cross_layer in self.cross_layers:
            cross_output = cross_layer(combined_input, cross_output)

        mlp_output = self.mlp(combined_input)

        X = torch.concat((cross_output, mlp_output), dim=1)

        X = self.concat_layer(X)
        return X
    def get_regularization_term(self, lmbd=0):
        total = 0
        for cross_layer in self.cross_layers:
            # sum the L2 norms of all the cross weights
            total += torch.norm(cross_layer.alphas, p=2)
        for mlp_layer in self.mlp:
            if isinstance(mlp_layer, nn.Linear):
                total += torch.norm(mlp_layer.weight, p=2)
        return lmbd * total