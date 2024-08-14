import torch
import torch.nn as nn

class NullPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'null preprocessor'

    def forward(self, j):
        return j