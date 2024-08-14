import torch
import torch.nn as nn

class BasePreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'base preprocessor'

    def forward(self, j):
        raise NotImplementedError('BasePreprocessor.forward() not Implemented')
