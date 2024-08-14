import torch
import torch.nn as nn

class BasePostprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'base postprocessor'

    def forward(self, j):
        raise NotImplementedError('BasePostprocessor.forward() not Implemented')
