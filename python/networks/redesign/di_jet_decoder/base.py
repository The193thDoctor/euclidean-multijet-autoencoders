# An embedder should make the jet in a larger dimension

import torch
import torch.nn as nn

class BaseDiJetDecoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.name = 'dijet_decoder_base'
    def forward(self, x):
        raise NotImplementedError("BaseDijetDecoder.forward() not Implemented")