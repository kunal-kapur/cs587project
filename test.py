import os
import sys

import unittest
import torch
import random
from models import CrossNet, DCN

class TestCrossNet(unittest.TestCase):
    def test_cross_net(self):
        torch.manual_seed(1)
        input = torch.tensor([[1,2,3]])

        cross_layer = CrossNet(3)

        cross_layer.alphas = torch.nn.Parameter((torch.ones(3, 1) * 2).float())
        cross_layer.bias = torch.nn.Parameter((torch.ones(3)).float())
        out = cross_layer.forward(input.float(), input.float())
        expected = torch.tensor([14,27,40])

        assert (torch.allclose(out.float(), expected.float()))
    
    def test_cross_net_complex(self):
        torch.manual_seed(1)
        input = torch.tensor([[1,2,3], [3,2,1]])

        cross_layer = CrossNet(3)

        cross_layer.alphas = torch.nn.Parameter((torch.ones(3, 1) * 2).float())
        cross_layer.bias = torch.nn.Parameter((torch.ones(3)).float())
        out = cross_layer.forward(input.float(), input.float())
        expected = torch.tensor([[14,27,40], [40,27,14]])

        assert (torch.allclose(out.float(), expected.float()))


class TestDCN(unittest.TestCase):
    def test_dcn_basic(self):
        torch.manual_seed(1)

        model = DCN(categorical_features=[4,5], num_numerical_features=2, embedding_dim=3,
                    dcn_layer_len=3, layer_sizes=[5,3], output_dim=1)

        cat_input = torch.tensor([[1,2,4,5,8,1]])
        num_input = torch.tensor([[5,6]])
        model(cat_input, num_input)


if __name__ == '__main__':
    unittest.main()