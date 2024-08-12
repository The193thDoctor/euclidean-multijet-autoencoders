import torch
from ...utils import FreezableModule

class BaseDiJetEncoder(FreezableModule):
    def __init__(self, dimension, depth, symmetrize, activation):
        super().__init__()
        self.dimension = dimension
        self.depth = depth
        self.symmetrize = symmetrize
        self.activation = activation
        self.name = 'base_di_jet_encoder'
    def forward(self, x):
        raise NotImplementedError("BaseDiJetEncoder.forward() not Implemented")