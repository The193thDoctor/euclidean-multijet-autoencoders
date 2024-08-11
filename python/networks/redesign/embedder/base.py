# An embedder should make the jet in a larger dimension

import torch
import torch.nn as nn

class BaseEmbedder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.name = 'base_embedder'

    def forward(self, x):
        raise NotImplementedError("BaseEmbedder.forward() not Implemented")

