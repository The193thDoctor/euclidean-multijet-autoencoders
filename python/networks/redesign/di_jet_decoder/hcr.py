import torch
import torch.nn as nn
from .base import BaseDiJetDecoder

class HCRDiJetDecoder(BaseDiJetDecoder):
    def __init__(self, dimension=20, depth=4, res_freq=1):
        super().__init__(dimension)
        self.name = f'hcr_di_jet_decoder_dim{dimension}_depth{depth}'

        self.depth = depth
        if depth < 2:
            raise ValueError('depth should be at least 1')

        self.initial_convolution = nn.ConvTranspose1d(self.dimension, self.dimension, 3, 3)
        self.jet_layers = nn.ModuleList()
        self.di_jet_layers = nn.ModuleList()
        for _ in range(1, self.depth-1):
            self.jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                         1, 1))
            self.di_jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                                3, 3))
        self.jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                         1, 1))
        self.di_jet_layers.append(nn.Conv1d(self.dimension, self.dimension,
                                            2, 3))

        self.initial_batch_norm = nn.BatchNorm1d(self.dimension)
        self.jet_batch_norm = nn.ModuleList()
        self.di_jet_batch_norm = nn.ModuleList()
        for _ in range(1, self.depth-1):
            self.jet_batch_norm.append(nn.BatchNorm1d(self.dimension))
            self.di_jet_batch_norm.append(nn.BatchNorm1d(self.dimension))
        self.jet_batch_norm.append(nn.BatchNorm1d(self.dimension))

        self.res_freq = res_freq

    def __extract(self, combined):
        batch_size = combined.shape[0]
        if __debug__:
            assert(combined.shape[1] == self.dimension)
        combined_grouped = combined.view(batch_size, self.dimension, -1, 3)
        jets = combined_grouped[:, :, :, 0:3].flatten(start_dim=2)
        di_jets = combined_grouped[:, :, :, 3]
        return jets, di_jets

    def forward(self, di_jets):
        if __debug__:
            input_shape = di_jets.shape
            output_shape = input_shape
            output_shape[2] *= 2

        jet_res, di_jet_res = 0, di_jets

        #initial convolution
        jet_combined = self.initial_batch_norm(self.initial_convolution(di_jets))
        jets, di_jets = self.__extract(jet_combined)
        if self.res_freq == 1:
            jets, di_jets = jets + jet_res, di_jets + di_jet_res
            jets, di_jets = torch.silu(jets), torch.silu(di_jets)
            jet_res, di_jet_res = jets, di_jets
        else:
            jets, di_jets = torch.silu(jets), torch.silu(di_jets)

        # layers
        for i in range(1, self.depth-1):
            jets = self.jet_layers[i-1](jets)
            combined_contribution = self.di_jet_batch_norm[i-1](self.di_jet_layers[i-1](di_jets))
            jet_contribution, di_jet_contribution = self.__extract(combined_contribution)
            jets, di_jets = jets + jet_contribution, di_jet_contribution
            jets, di_jets = self.jet_batch_norm[i-1](jets), self.di_jet_batch_norm[i-1](di_jets)
            if (i+1) % self.res_freq == 0:
                jets, di_jets = jets + jet_res, di_jets + di_jet_res
                jets, di_jets = torch.silu(jets), torch.silu(di_jets)
                jet_res, di_jet_res = jets, di_jets
            else:
                jets, di_jets = torch.silu(jets), torch.silu(di_jets)

        # final layer, di_jet is dropped so it is different
        jets = self.jet_layers[self.depth-2](jets)
        combined_contribution = self.di_jet_batch_norm[self.dpeth-2](self.di_jet_layers[self.depth-2](di_jets))
        jet_contribution, di_jet_contribution = self.__extract(combined_contribution)
        jets, di_jets = jets + jet_contribution, di_jet_contribution
        if __debug__:
            assert (torch.all(torch.isclose(di_jets, torch.tensor(0.0), atol=1e-6)))
        jets = self.jet_batch_norm[self.depth-2](jets)
        if self.depth % self.res_freq == 0:
            jets = jets + jet_res
        jets = torch.silu(jets)

        if __debug__:
            assert(jets.shape == output_shape)

        return jets
