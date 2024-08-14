import torch
import torch.nn as nn

class NullPostprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'null postprocessor'

    def forward(self, j):
        return j