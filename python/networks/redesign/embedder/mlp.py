import torch
import torch.nn as nn
from .base import EmbedderBase

class EmbedderMLP(EmbedderBase):
    def __init__(self, dimension=20, depth=4, dropout=0.3, res_freq=1):
        super().__init__(dimension)

        self.name = 'mlp_embedder_dim{}'.format(dimension)

        self.depth = depth
        if depth < 1:
            raise ValueError('depth should be at least 1')
        dim = 4
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            self.layers.append(nn.Linear(dim, self.dimension))
            dim = self.dimension

        self.batch_norm = nn.ModuleList()
        for _ in range(self.depth):
            self.batch_norm.append(nn.BatchNorm1d(self.dimension))

        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        self.res_freq = res_freq

    def forward(self, x):
        x_res = None
        batch_size = x.shape[0]
        for i in range(self.depth):
            x = x.swapaxes(1, 2)
            x = self.layers[i](x) # nn.Linear acts on the last dimension
            x = x.swapaxes(1, 2)

            x = self.batch_norm[i](x)

            if i % self.res_freq == 0:
                if x_res is not None:
                    x = x + x_res
                    x = torch.silu(x)
                x_res = x
            else:
                x = torch.silu(x)

        x = self.drop(x)
        assert x.shape == (batch_size, self.dimension, 4)
        return x

    


