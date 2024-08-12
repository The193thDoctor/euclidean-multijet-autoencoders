import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseEmbedder

class MLPEmbedder(BaseEmbedder):
    def __init__(self, dimension=20, depth=4, activation=F.silu, dropout=0.3, res_len=1):
        super().__init__(dimension, depth, activation)
        self.name = f'mlp_embedder_dim{dimension}'

        if self.depth < 1:
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

        self.res_len = res_len # there is one residual connection every res_len layers

    def forward(self, x):
        x_res = x
        if __debug__:
            batch_size = x.shape[0]
        for i in range(self.depth):
            x = x.swapaxes(1, 2)
            x = self.layers[i](x) # nn.Linear acts on the last dimension
            x = x.swapaxes(1, 2)

            x = self.batch_norm[i](x)

            if (i+1) % self.res_len == 0:
                x = x + x_res
                x = self.activation(x)
                x_res = x
            else:
                x = self.activation(x)

        x = self.drop(x)
        if __debug__:
            assert x.shape == (batch_size, self.dimension, 4)
        return x

    


