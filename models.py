mport torch
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
        R"""
        image_size: int, size of the input image, For MNIST it is 28
        output_dim: int, size of the output layer, For MNIST it is 10(equal to number of classes)
        """
        self.alphas = nn.Parameter(torch.Tensor(input_dim, 1, 1))
        self.bias = torch.nn.Parameter(torch.Tensor(1))
    
    def to(self, device):
        self.alphas = self.alphas.to(device)
        self.bias = self.bias.to(device)
        return super().to(device)
    def forward(self, X):
        # Input shape: torch.Size([minibatch, input_dim])
        ####
        # Task: implement the forward pass
        # for equivarinat MLP
        # return the output of the layer 
        ####
        X = torch.mm(X, X.T)
        return nn.functional.linear(X, self.alphas, self.bias)

class DCN(nn.Module):
    R"""
    image_size: int, size of the input image, For MNIST it is 28
    layer_sizes: List[int], size of the hidden layers
    output_dim: int, size of the output layer, For MNIST it is 10(equal to number of classes)
    """
    def __init__(self, input_size, dcn_layer_size, layer_sizes, output_dim):
        super().__init__()
        self.image_size = image_size
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        layers = []
        ####
        # Task: initialize the linear layers
        ###
        prev_size = self.image_size * self.image_size
        for i in range(0, len(layer_sizes)):
            curr_layer = torch.nn.Linear(in_features=prev_size, out_features=layer_sizes[i])
            layers.append(curr_layer)
            layers.append(torch.nn.ReLU())
            prev_size = layer_sizes[i]
        layers.append(torch.nn.Linear(in_features=prev_size, out_features=output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def get_hidden_rep(self, X):
        hidden_rep = []
        X = X.reshape(-1, self.image_size * self.image_size)
        for layer in self.mlp:
            X = layer(X)
            hidden_rep.append(X.clone())
        return hidden_rep

    def forward(self, X):
        X = X.reshape(-1, self.image_size * self.image_size)
        ####
        # Task: implement the forward pass
        ###
        out = self.mlp.forward(X)
        return out
