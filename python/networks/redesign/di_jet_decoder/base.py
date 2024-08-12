import torch
from ...utils import FreezableModule

class BaseDiJetDecoder(FreezableModule):
    def __init__(self, dimension, depth, activation):
        super().__init__()
        self.dimension = dimension
        self.depth = depth
        self.activation = activation
        self.name = 'base_di_jet_decoder'
    def forward(self, x):
        raise NotImplementedError("BaseDiJetDecoder.forward() not Implemented")