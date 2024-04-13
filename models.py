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

        bias = self.bias + X
        X = X.unsqueeze(2)
        input = input.unsqueeze(2)

        transposed_X = torch.transpose(X, 1, 2) # keep batch the same
        X = torch.bmm(X, transposed_X)

        val = torch.matmul(X, self.alphas).squeeze(2)

        return val + bias


class DCN(nn.Module):
    R"""
    input_size: int, size of the input
    dcn_layer_len: number of cross layers we have
    layer_sizes: List[int], size of the hidden layers in the hidden network
    output_dim: int, size of the output layer
    embedding_dim: dimension size of the embedding, typically the size of the input dim
    """
    def __init__(self, categorical_features, num_numerical_features, 
                 embedding_dim, dcn_layer_len, layer_sizes, output_dim):
        

        super().__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) for num_categories in categorical_features
        ])

        # final input dimension
        input_dim = len(categorical_features)  * embedding_dim + num_numerical_features

        self.cross_layers = [CrossNet(input_dim=input_dim) for i in range(dcn_layer_len)]

        prev_dim = input_dim
        mlp = []
        for i in range(0, len(layer_sizes)):
            mlp.append(torch.nn.Linear(in_features=prev_dim, out_features=layer_sizes[i]))
            mlp.append(torch.nn.ReLU())
            prev_dim = layer_sizes[i]

        mlp.pop() # Pop last ReLU layer?
        self.mlp = torch.nn.Sequential(*mlp)

        # the cross layers have input dim and we add to last layer size
        concat_layer = \
        [torch.nn.Linear(in_features= input_dim + layer_sizes[-1], out_features=output_dim), torch.nn.Sigmoid()]

        self.concat_layer = torch.nn.Sequential(*concat_layer)

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



        for cross_layer in self.cross_layers:
            cross_output = cross_layer(combined_input, cross_output)

        mlp_output = self.mlp(combined_input)

        X = torch.concat((cross_output, mlp_output), dim=1)
        X = self.concat_layer(X)
        return X
