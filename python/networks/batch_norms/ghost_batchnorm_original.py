import torch
from torch import nn as nn

from networks.utils import vector_print

class Ghost_Batch_Norm(nn.Module):  # https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition.
    def __init__(self, features, ghost_batch_size=32, number_of_ghost_batches=64, n_averaging=1, stride=1, eta=0.9,
                 bias=True, name='', conv=False, conv_transpose=False, features_out=None):  # number_of_ghost_batches was initially set to 64
        super(Ghost_Batch_Norm, self).__init__()
        self.name = name
        self.index = None
        self.stride = stride if not conv_transpose else 1
        self.features = features
        self.features_out = features_out if features_out is not None else self.features
        self.register_buffer('ghost_batch_size', torch.tensor(ghost_batch_size, dtype=torch.long))
        self.register_buffer('n_ghost_batches', torch.tensor(number_of_ghost_batches * n_averaging, dtype=torch.long))
        self.conv = False
        self.gamma = None
        self.bias = None
        self.updates = 0
        if conv:
            self.conv = nn.Conv1d(self.features, self.features_out, stride, stride=stride, bias=bias)
        elif conv_transpose:
            self.conv = nn.ConvTranspose1d(self.features, self.features_out, stride, stride=stride, bias=bias)
        else:
            self.gamma = nn.Parameter(torch.ones(self.features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.features))
        self.running_stats = True
        self.initialized = False

        self.register_buffer('eps', torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer('eta', torch.tensor(eta, dtype=torch.float))
        self.register_buffer('m', torch.zeros((1, 1, self.stride, self.features), dtype=torch.float))
        self.register_buffer('s', torch.ones((1, 1, self.stride, self.features), dtype=torch.float))
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float))
        self.register_buffer('two', torch.tensor(2., dtype=torch.float))
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

    @torch.no_grad()
    def set_mean_std(self, x):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        pixel_groups = pixels // self.stride
        x = x.detach().transpose(1, 2).contiguous().view(batch_size * pixels, 1, self.features)
        # this won't work for any layers with stride!=1
        x = x.view(-1, 1, self.stride, self.features)
        m64 = x.mean(dim=0, keepdim=True, dtype=torch.float64)
        self.m = m64.type(torch.float32)
        self.s = x.std(dim=0, keepdim=True)
        self.initialized = True
        self.running_stats = False
        self.print()

    def set_ghost_batches(self, n_ghost_batches):
        self.register_buffer('n_ghost_batches', torch.tensor(n_ghost_batches, dtype=torch.long))

    def forward(self, x, debug=False):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        pixel_groups = pixels // self.stride

        if self.training and self.n_ghost_batches != 0:
            # this has been changed from self.ghost_batch_size = batch_size // self.n_ghost_batches.abs()
            self.ghost_batch_size = torch.div(batch_size, self.n_ghost_batches.abs(), rounding_mode='trunc')

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = x.transpose(1, 2).contiguous().view(self.n_ghost_batches.abs(), self.ghost_batch_size * pixel_groups,
                                                    self.stride, self.features)

            gbm = x.mean(dim=1, keepdim=True)
            gbs = (x.var(dim=1, keepdim=True) + self.eps).sqrt()

            #
            # Keep track of running mean and standard deviation.
            #
            if self.running_stats or debug:
                # Use mean over ghost batches for running mean and std
                bm = gbm.detach().mean(dim=0, keepdim=True)
                bs = gbs.detach().mean(dim=0, keepdim=True)

                if debug and self.initialized:
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

            if self.running_stats:
                # Simplest possible method

                if self.initialized:
                    self.m = self.eta * self.m + (self.one - self.eta) * bm
                    self.s = self.eta * self.s + (self.one - self.eta) * bs
                else:
                    self.m = self.zero * self.m + bm
                    self.s = self.zero * self.s + bs
                    self.initialized = True

            if self.n_ghost_batches > 0:
                x = x - gbm
                x = x / gbs
            else:
                x = x.view(batch_size, pixel_groups, self.stride, self.features)
                x = x - self.m
                x = x / self.s

        else:

            # Use mean and standard deviation buffers rather than batch statistics
            # .view(self.n_ghost_batches, self.ghost_batch_size*pixel_groups, self.stride, self.features)
            x = x.transpose(1, 2).view(batch_size, pixel_groups, self.stride, self.features)
            x = x - self.m
            x = x / self.s

        if self.conv:
            # back to standard indexing for convolutions: [batch, feature, pixel]
            x = x.view(batch_size, pixels, self.features).transpose(1, 2).contiguous()
            x = self.conv(x)
        else:
            x = x * self.gamma
            if self.bias is not None:
                x = x + self.bias
            # back to standard indexing for convolutions: [batch, feature, pixel]
            x = x.view(batch_size, pixels, self.features).transpose(1, 2).contiguous()

        return x

    #
