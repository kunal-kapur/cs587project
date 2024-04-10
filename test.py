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

        cross_layer.alphas = torch.nn.Parameter((torch.ones_like(input) * 2).float())
        cross_layer.bias = torch.nn.Parameter((torch.ones_like(input)).float())
        out = cross_layer.forward(input.float())
        expected = torch.tensor([14,27,40])
        print(expected, out)
        assert (torch.allclose(out.float(), expected.float()))


if __name__ == '__main__':
    unittest.main()