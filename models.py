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

        self.alphas = nn.Parameter(torch.Tensor(1, input_dim, 1))
        self.bias = torch.nn.Parameter(torch.Tensor(input_dim))

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
        # Input shape: torch.Size([minibatch, input_dim])
        bias = self.bias + X
        X = X.unsqueeze(2)

        transposed_X = torch.transpose(X, 1, 2) # keep batch the same
        X = torch.bmm(X, transposed_X)
        print("X", X)
        print("wewight", self.alphas)
        val = torch.bmm(X, self.alphas.unsqueeze(2)).squeeze(2)
        print(val)
        print(bias)
        return val + bias
        return nn.functional.linear(X, self.alphas, bias)

class DCN(nn.Module):
    R"""
    input_size: int, size of the input
    dcn_layer_len: number of cross layers we have
    layer_sizes: List[int], size of the hidden layers in the hidden network
    output_dim: int, size of the output layer
    embedding_dim: dimension size of the embedding, typically the size of the input dim
    """
    def __init__(self, num_categorical_features, num_numerical_features, 
                 embedding_dim, dcn_layer_len, layer_sizes, output_dim):
        

        super().__init__()
        self.embedding_layer = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical_features
        ])

        # find sum of total embedding size
        total_embedding_size = sum(num_categories * embedding_dim for num_categories in num_categorical_features)

        # final input dimension
        input_dim = total_embedding_size + num_numerical_features

        self.cross_layers = [CrossNet(input_dim=input_dim) for i in range(dcn_layer_len)]

        prev_dim = input_dim
        self.mlp = []
        for i in range(0, len(layer_sizes)):
            self.mlp.append(torch.nn.Linear(in_features=prev_dim, out_features=layer_sizes[i]))
            self.mlp.append(torch.nn.ReLU())

        # Pop last ReLU layer?
        self.mlp = torch.nn.Sequential(self.mlp)
        self.cross_layers = torch.nn.Sequential(self.cross_layers)

        # the cross layers have input dim and we add to last layer size
        self.concat_layer = \
        [torch.nn.Linear(in_features= input_dim + layer_sizes[-1], out_features=output_dim), torch.nn.Sigmoid()]

        self.concat_layer = torch.nn.Sequential(self.concat_layer)

    def forward(self, categorical_input, numerical_input):
        embedded_categorical = [
            embedding_layer(categorical_input[:, i]) for i, embedding_layer in enumerate(self.embedding_layer)
        ]

        embedded_categorical = torch.cat(embedded_categorical, dim=1)  # Concatenate along feature dimension
        combined_input = torch.cat([embedded_categorical, numerical_input], dim=1)

        cross_output = self.cross_layers(X)
        mlp_output = self.mlp(X)
        X = torch.concat(cross_output, mlp_output, dim=1)
        X = self.concat_layer(X)
        return X
