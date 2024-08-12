import torch
import torch.nn as nn
from .base import BaseDiJetEncoder

class HCRDiJetEncoder(BaseDiJetEncoder):
    def __init__(self, dimension=20, depth=4, res_freq=1, symmetrize=True):
        super().__init__(dimension)
        self.name = f'hcr_di_jet_encoder_dim{dimension}_depth{depth}'

        self.depth = depth
        if depth < 2:
            raise ValueError('depth should be at least 1')

        self.jet_layers = nn.ModuleList()
        self.di_jet_layers = nn.ModuleList()
        for _ in range(self.depth-1):
            self.jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                         1, 1))
            self.di_jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                                3, 3))
        self.final_convolution = nn.Conv1d(self.dimension, self.dimension, 3, 3)

        self.jet_batch_norm = nn.ModuleList()
        self.di_jet_batch_norm = nn.ModuleList()
        for _ in range(self.depth-1):
            self.jet_batch_norm.append(nn.BatchNorm1d(self.dimension))
            self.di_jet_batch_norm.append(nn.BatchNorm1d(self.dimension))
        self.final_batch_norm = nn.BatchNorm1d(self.dimension)

        self.res_freq = res_freq

        self.symmetrize = symmetrize

    def __interleave(self, jets, di_jets):
        batch_size = jets.shape[0]
        if __debug__:
            assert(batch_size == di_jets.shape[0])
            assert(jets.shape[1] == di_jets.shape[1])
        jets_grouped = jets.view(batch_size, self.dimension, -1, 2)
        di_jets_grouped = di_jets.view(batch_size, self.dimension, -1, 1)
        combined_grouped = torch.cat((jets, di_jets), dim=-1)
        combined = combined_grouped.flatten(start_dim=2)
        return combined

    def forward(self, jets):
        batch_size = jets.shape[0]

        if __debug__:
            input_shape = jets.shape
            output_shape = input_shape
            output_shape[2] //= 2


        jets_grouped = jets.view(batch_size, self.dimension, -1, 2)
        if self.symmetrize:
            jets_reversed = jets_grouped.flip(dims=[-1]).flatten(start_dim=2)
            jets_grouped = torch.cat((jets, jets_reversed), dim=2)
            jets = jets_grouped.flatten(start_dim=2)
        di_jets = jets_grouped.sum(dim=-1)
        jet_res, di_jet_res = jets, di_jets

        # layers
        for i in range(0, self.depth-1):
            jet_combined = self.__interleave(jets, di_jets)
            di_jets = self.di_jet_batch_norm[i](self.di_jet_layers[i](jet_combined))
            jets = self.jet_batch_norm[i](self.jet_layers[i](jets))
            if (i+1) % self.res_freq == 0:
                jets, di_jets = jets + jet_res, di_jets + di_jet_res
                jets, di_jets = torch.silu(jets), torch.silu(di_jets)
                jet_res, di_jet_res = jets, di_jets
            else:
                jets, di_jets = torch.silu(jets), torch.silu(di_jets)

        # final convolution
        jet_combined = self.__interleave(jets, di_jets)
        result = self.final_batch_norm(self.final_convolution(jet_combined))
        if self.depth % self.res_freq == 0:
            result = result + di_jet_res
        result = torch.silu(result)

        # symmetrize
        if self.symmetrize:
            result = result.view(batch_size, self.dimension, 2, -1).mean(dim=2)

        if __debug__:
            assert(result.shape == output_shape)

        return result
