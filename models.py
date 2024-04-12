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

        self.alphas = torch.nn.Parameter(torch.Tensor(input_dim, 1))
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
    def forward(self, initial, X):
        # Input shape: torch.Size([minibatch, input_dim])
        # fix it

        bias = self.bias + X

        # add an extra dimension for matrix multiplication
        X = X.unsqueeze(2)
        initial = initial.unsqueeze(2)

        transposed_X = torch.transpose(X, 1, 2) # transposing with respect to batch
        X = torch.bmm(initial, transposed_X)
        val = torch.matmul(X, self.alphas).squeeze(2)
        return val + bias

class DCN(nn.Module):
    
    def __init__(self, categorical_features: list, num_numerical_features: int, 
                 embedding_dim: int, dcn_layer_len: int, layer_sizes: int, output_dim: int):
        """_summary_

        Args:
            categorical_features (list): list of unique categories for each categorical feature
            num_numerical_features (int): number of numeriocal features
            embedding_dim (int): dimension for the embedding layer
            dcn_layer_len (int): number of cross layers
            layer_sizes (int): number of layers in the normal mlp
            output_dim (int): output dimension we want
        """
        
        super().__init__()

        self.num_categorical_features = len(categorical_features)
        self.num_numerical_features = num_numerical_features
        self.embedding_dim = embedding_dim
        self.dcn_layer_len = dcn_layer_len
        self.output_dim = output_dim

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) for num_categories in categorical_features
        ])


        # final input dimension
        input_dim = self.num_categorical_features * embedding_dim + num_numerical_features
        #print("input dim", input_dim, total_embedding_size, num_numerical_features)
        self.cross_layers = [CrossNet(input_dim=input_dim) for i in range(dcn_layer_len)]

        prev_dim = input_dim
        mlp = []
        for i in layer_sizes:
            mlp.append(torch.nn.Linear(in_features=prev_dim, out_features=i))
            mlp.append(torch.nn.ReLU())
            prev_dim = i
        # Pop last ReLU layer?
        mlp.pop()
        #print(mlp)
        self.mlp = torch.nn.Sequential(*mlp)

        # the cross layers have input dim and we add to last layer size
        concat_layer = \
        [torch.nn.Linear(in_features= input_dim + layer_sizes[-1], out_features=output_dim), torch.nn.Sigmoid()]

        self.concat_layer = torch.nn.Sequential(*concat_layer)

    def forward(self, categorical_input, numerical_input):
        """
        Args:
            categorical_input (_type_): the category inputs
            numerical_input (_type_): the separate inputs for the numerical features
        """
        
        if categorical_input.shape[1] != self.num_categorical_features:
            raise Exception("Categorical dimension must match input")
        if numerical_input.shape[1] != self.num_numerical_features:
            raise Exception("Numerical dimension must match input")


        #print(self.embedding_layers[0](categorical_input[:, 0]))

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
