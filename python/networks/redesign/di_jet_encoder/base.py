import torch
import torch.nn as nn

class BaseDiJetEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.name = 'base_di_jet_encoder'
    def forward(self, x):
        raise NotImplementedError("BaseDiJetEncoder.forward() not Implemented")