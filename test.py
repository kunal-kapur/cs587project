import os
import sys

import unittest
import torch
import random
from models import CrossNet, DCN

class TestCrossNet(unittest.TestCase):
    def test_cross_net(self):
        '''Testing to account for value'''
        torch.manual_seed(1)
        input = torch.tensor([[1,2,3]])
        cross_layer = CrossNet(3)
        cross_layer.alphas = torch.nn.Parameter((torch.ones(1,3,1) * 2).float())
        cross_layer.bias = torch.nn.Parameter((torch.ones(3)).float())
        out = cross_layer.forward(input.float(), input.float())
        expected = torch.tensor([[14,27,40]])
        assert (torch.allclose(out.float(), expected.float()))

    def test_cross_net_complext(self):
        '''Testing to account for several batches'''
        torch.manual_seed(1)
        input = torch.tensor([[1,2,3],[3,2,1]])
        cross_layer = CrossNet(3)
        cross_layer.alphas = torch.nn.Parameter((torch.ones(1,3,1) * 2).float())
        cross_layer.bias = torch.nn.Parameter((torch.ones(3)).float())
        out = cross_layer.forward(input.float(), input.float())
        expected = torch.tensor([[14,27,40], [40,27,14]])
        assert (torch.allclose(out.float(), expected.float()))

class TestDCN(unittest.TestCase):

    def test_DCN_simple(self):
        '''
        Create a generic DCN model. Simply check that it runs
        '''
        # TODO 
        model = DCN(categorical_features=[3,5], num_numerical_features=5, 
                    embedding_dim=4, dcn_layer_len=4, layer_sizes=[2,3,3], output_dim=2)
        cat_input = torch.tensor([[2,1]])
        num_input = torch.tensor([[1,2,3,4,5]])
        model(cat_input, num_input)
        


if __name__ == '__main__':
    unittest.main()