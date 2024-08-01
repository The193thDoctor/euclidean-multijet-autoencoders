import torch
import torch.nn as nn

class PreprocessBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError("PreprocessBase forward not Implemented")
