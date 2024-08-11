import torch
import torch.nn as nn
from .base import DijetEncoderBase

class DijetEncoderHCR(DijetEncoderBase):
    def __init__(self, dimension=20, depth=4, dropout=0.3, res_freq=1, symmetrize=True):
        super().__init__(dimension)

        self.depth = depth
        if depth < 1:
            raise ValueError('depth should be at least 1')
        dim = 4
        self.jet_layers = nn.ModuleList()
        self.di_jet_layers = nn.ModuleList()
        for _ in range(self.depth):
            self.layers.append(nn.Conv1d(self.dimension, self.dimension,
                                         2, 2))
            self.di_jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                                3, 3))
        self.final_convolution = nn.Conv1d(self.dimension, self.dimension, 3, 3)

        self.jet_batch_norm = nn.ModuleList()
        self.di_jet_batch_norm = nn.ModuleList()
        for _ in range(self.depth):
            self.jet_batch_norm.append(nn.BatchNorm1d(self.dimension))
            self.di_jet_batch_norm.append(nn.BatchNorm1d(self.dimension))
        self.final_batch_norm = nn.BatchNorm1d(self.dimension)

        self.res_freq = res_freq

        self.symmetrize = symmetrize

    def forward(self, jets):
        batch_size = jets.shape[0]

        # I will group together jets and used the grouped version
        jets = jets.view(batch_size, self.dimension, -1, 2)
        if self.symmetrize:
            jets_reversed = jets.flip(dims=[-1]).flatten(2, 3)
            jets = torch.cat((jets, jets_reversed), dim=2)
        di_jets = jets.sum(dim=-1, keepdim=True)
        jet_res, di_jet_res = jets, di_jets

        # layers
        for i in range(self.depth):
            jet_combined_flatten = torch.cat((jets, di_jets), dim=-1).flatten(start_dim=2)
            di_jet_flatten = self.di_jet_batch_norm[i](self.di_jet_layers[i](jet_combined_flatten))
            di_jets = di_jet_flatten.view(batch_size, self.dimension, -1, 1)
            jets = self.jet_batch_norm[i](self.jet_layers[i](jet_combined_flatten))
            jets = jets.view(batch_size, self.dimension, -1, 2)
            if i % self.res_freq == 0:
                di_jets = di_jets + di_jet_res
                jets = torch.silu(jets + jet_res)
                jet_res, di_jet_res = jets, di_jets
            else:
                jets, di_jets = torch.silu(jets), torch.silu(di_jets)

        # final convolution
        jet_combined_flatten = torch.cat((jets, di_jets), dim=-1).flatten(start_dim=2)
        ## convert directly to final shape
        result = self.final_batch_norm(self.final_convolution(jet_combined_flatten))
        if self.symmetrize:
            result = result.view(batch_size, self.dimension, 2, -1).mean(dim=2)
        return result
