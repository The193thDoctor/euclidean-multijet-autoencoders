import torch
from torch import nn as nn

from networks.utils import vector_print


class Ghost_Batch_Norm(nn.Module):  # https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition.
    def __init__(self, features, ghost_batch_size=32, number_of_ghost_batches=64, n_averaging=1, eta=0.9,
                 bias=True):  # number_of_ghost_batches was initially set to 64
        super(Ghost_Batch_Norm, self).__init__()
        self.index = None
        self.features = features
        self.register_buffer('ghost_batch_size', torch.tensor(ghost_batch_size, dtype=torch.long))
        self.register_buffer('n_ghost_batches', torch.tensor(number_of_ghost_batches * n_averaging, dtype=torch.long))
        self.conv = False
        self.gamma = nn.Parameter(torch.ones(self.features))
        self.bias = nn.Parameter(torch.zeros(self.features))

        self.register_buffer('eps', torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer('eta', torch.tensor(eta, dtype=torch.float))
        self.register_buffer('m', None)
        self.register_buffer('s', None)

    def print(self):
        print('-' * 50)
        print(self.name)
        for i in range(self.stride):
            print(" mean ", end='')
            vector_print(self.m[0, 0, i, :])
        for i in range(self.stride):
            print("  std ", end='')
            vector_print(self.s[0, 0, i, :])
        if self.gamma is not None:
            print("gamma ", end='')
            vector_print(self.gamma.data)
            if self.bias is not None:
                print(" bias ", end='')
                vector_print(self.bias.data)
        print()

    def set_ghost_batches(self, n_ghost_batches):
        self.register_buffer('n_ghost_batches', torch.tensor(n_ghost_batches, dtype=torch.long))

    def forward(self, x, debug=False):
        batch_size = x.shape[0]
        remaining_dim = x.shape[2:]

        if self.training and self.n_ghost_batches != 0:
            # this has been changed from self.ghost_batch_size = batch_size // self.n_ghost_batches.abs()
            self.ghost_batch_size = torch.div(batch_size, self.n_ghost_batches.abs(), rounding_mode='trunc')

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = x.view(self.n_ghost_batches.abs(), self.ghost_batch_size, self.features, *remaining_dim)


            gbm = x.mean(dim=1, keepdim=True)
            gbs = (x.var(dim=1, keepdim=True) + self.eps).sqrt()

            #
            # Keep track of running mean and standard deviation.
            #
            # Use mean over ghost batches for running mean and std
            bm = gbm.detach().mean(dim=0, keepdim=True)
            bs = gbs.detach().mean(dim=0, keepdim=True)

            if debug:
                gbms = gbm.detach().std(dim=0, keepdim=True)
                gbss = gbs.detach().std(dim=0, keepdim=True)
                m_pulls = (bm - self.m) / gbms
                s_pulls = (bs - self.s) / gbss
                # s_ratio = bs/self.s
                # if (m_pulls.abs()>5).any() or (s_pulls.abs()>5).any():
                print()
                print(self.name)
                print('self.m\n', self.m)
                print('    bm\n', bm)
                print('  gbms\n', gbms)
                print('m_pulls\n', m_pulls, m_pulls.abs().mean(), m_pulls.abs().max())
                print('-------------------------')
                print('self.s\n', self.s)
                print('    bs\n', bs)
                print('  gbss\n', gbss)
                print('s_pulls\n', s_pulls, s_pulls.abs().mean(), s_pulls.abs().max())
                # print('s_ratio\n',s_ratio)
                print()
                # input()

            if self.m is not None:
                self.m = self.eta * self.m + (1.0 - self.eta) * bm
                self.s = self.eta * self.s + (1.0 - self.eta) * bs
            else:
                self.m = bm
                self.s = bs

            x = (x - gbm) / gbs

        else:
            x = (x - self.m) / self.s

        x = x.view(batch_size, self.features, *remaining_dim)
        num_remaining_dims = len(remaining_dim)
        dummy_dims = torch.concat((torch.tensor([self.features]),
            torch.ones(num_remaining_dims, dtype=torch.long)), 0) # temporary dimension for broadcasting
        x = x * self.gamma.view(*dummy_dims) + self.bias.view(*dummy_dims)
        # back to standard indexing for convolutions: [batch, feature, pixel]
        return x
