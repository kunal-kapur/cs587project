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

        self.alphas = nn.Parameter(torch.Tensor(input_dim, 1, 1))
        self.bias = torch.nn.Parameter(torch.Tensor(1))

        # Kaiming initialization
        stdv = 1.0/math.sqrt(input_dim)
        self.alphas.data.uniform_(-stdv, stdv)
        self.alphas = self.alphas
        self.bias.data.zero_()
    
    def to(self, device):
        self.alphas = self.alphas.to(device)
        self.bias = self.bias.to(device)
        return super().to(device)
    def forward(self, X):

        X = torch.mm(X, X.T)
        return nn.functional.linear(X, self.alphas, self.bias)

class DCN(nn.Module):
    R"""
    input_size: int, size of the input
    dcn_layer_len: number of cross layers we have
    layer_sizes: List[int], size of the hidden layers
    output_dim: int, size of the output layer
    embedding_dim: dimension size of the embedding, typically the size of the input dim
    """
    def __init__(self, input_size, dcn_layer_len, layer_sizes, output_dim, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=self.embedding_dim)

        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        self.cross_layers = [CrossNet(input_dim=input_size) for i in range(dcn_layer_len)]

        prev_dim = layer_sizes[0]
        self.mlp = []
        for i in range(1, len(layer_sizes)):
            self.mlp.append(torch.nn.Linear(in_features=prev_dim, out_features=layer_sizes[i]))
            self.mlp.append(torch.nn.ReLU())

        # Pop last ReLU layer?
        self.mlp = torch.nn.Sequential(self.mlp)
        self.cross_layers = torch.nn.Sequential(self.cross_layers)
        self.concat_layer = \
        [torch.nn.Linear(in_features= input_size + layer_sizes[-1], out_features=output_dim), torch.nn.Sigmoid()]

        self.concat_layer = torch.nn.Sequential(self.concat_layer)

    def forward(self, X):
        X = self.embedding(X)
        cross_output = self.cross_layers(X)
        mlp_output = self.mlp(X)
        X = torch.concat(cross_output, mlp_output, dim=1)
        X = self.concat_layer(X)
        return X
