import torch
import torch.nn as nn

class BaseEmbedder(nn.Module):
    def __init__(self, dimension, depth, activation):
        super().__init__()
        self.dimension = dimension
        self.depth = depth
        self.activation = activation
        self.name = 'base_embedder'

    def forward(self, x):
        raise NotImplementedError("BaseEmbedder.forward() not Implemented")

