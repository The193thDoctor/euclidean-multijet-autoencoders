# An embedder should make the jet in a larger dimension

import torch
import torch.nn as nn

class DijetEncoderBase(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.name = 'base_dijet_encoder'
    def forward(self, x):
        raise NotImplementedError("EmbedderBase forward not Implemented")