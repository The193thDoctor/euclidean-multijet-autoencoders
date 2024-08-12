import torch
import torch.nn as nn

class BaseDiJetDecoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.name = 'di_jet_decoder_base'
    def forward(self, x):
        raise NotImplementedError("BaseDiJetDecoder.forward() not Implemented")