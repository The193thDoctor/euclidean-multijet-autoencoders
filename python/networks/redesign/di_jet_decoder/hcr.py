import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDiJetDecoder

class HCRDiJetDecoder(BaseDiJetDecoder):
    def __init__(self, dim=20, depth=4, activation = F.silu, res_len=1):
        super().__init__(dim, depth, activation)
        self.name = f'hcr_di_jet_decoder_dim{self.dim}_depth{self.depth}'

        if depth < 2:
            raise ValueError('depth should be at least 1')

        # transpose convolution layers
        self.initial_convolution = nn.ConvTranspose1d(self.dim, self.dim, 3, 3)
        self.jet_layers = nn.ModuleList()
        self.di_jet_layers = nn.ModuleList()
        for _ in range(1, self.depth-1):
            self.jet_layers.append(nn.ConvTranspose1d(self.dim, self.dim, 1, 1))
            self.di_jet_layers.append(nn.ConvTranspose1d(self.dim, self.dim,3, 3))
        self.jet_layers.append(nn.ConvTranspose1d(self.dim, self.dim,1, 1))
        self.di_jet_layers.append(nn.ConvTranspose1d(self.dim, self.dim,2, 3))

        # batch normalization layers
        self.initial_batch_norm = nn.BatchNorm1d(self.dim)
        self.jet_batch_norm = nn.ModuleList()
        self.di_jet_batch_norm = nn.ModuleList()
        for _ in range(1, self.depth-1):
            self.jet_batch_norm.append(nn.BatchNorm1d(self.dim))
            self.di_jet_batch_norm.append(nn.BatchNorm1d(self.dim))
        self.jet_batch_norm.append(nn.BatchNorm1d(self.dim))

        self.res_len = res_len # there is one residual connection every res_len layers

    def __extract(self, combined):
        batch_size = combined.shape[0]
        if __debug__:
            assert(combined.shape[1] == self.dim)
        combined_grouped = combined.view(batch_size, self.dim, -1, 3)
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
        if self.res_len == 1:
            jets, di_jets = jets + jet_res, di_jets + di_jet_res
            jets, di_jets = self.activation(jets), self.activation(di_jets)
            jet_res, di_jet_res = jets, di_jets
        else:
            jets, di_jets = self.activation(jets), self.activation(di_jets)

        # layers
        for i in range(1, self.depth-1):
            jets = self.jet_layers[i-1](jets)
            combined_contribution = self.di_jet_batch_norm[i-1](self.di_jet_layers[i-1](di_jets))
            jet_contribution, di_jet_contribution = self.__extract(combined_contribution)
            jets, di_jets = jets + jet_contribution, di_jet_contribution
            jets, di_jets = self.jet_batch_norm[i-1](jets), self.di_jet_batch_norm[i-1](di_jets)
            if (i+1) % self.res_len == 0:
                jets, di_jets = jets + jet_res, di_jets + di_jet_res
                jets, di_jets = self.activation(jets), self.activation(di_jets)
                jet_res, di_jet_res = jets, di_jets
            else:
                jets, di_jets = self.activation(jets), self.activation(di_jets)

        # final layer, di_jet is dropped so it is different
        jets = self.jet_layers[self.depth-2](jets)
        combined_contribution = self.di_jet_batch_norm[self.dpeth-2](self.di_jet_layers[self.depth-2](di_jets))
        jet_contribution, di_jet_contribution = self.__extract(combined_contribution)
        jets, di_jets = jets + jet_contribution, di_jet_contribution
        if __debug__:
            assert (torch.all(torch.isclose(di_jets, torch.tensor(0.0), atol=1e-6)))
        jets = self.jet_batch_norm[self.depth-2](jets)
        if self.depth % self.res_len == 0:
            jets = jets + jet_res
        jets = self.activation(jets)

        if __debug__:
            assert(jets.shape == output_shape)

        return jets
