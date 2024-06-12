import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

torch.manual_seed(0)  # make training results repeatable


def vector_print(vector, end='\n'):
    vectorString = ", ".join([f'{element:7.2f}' for element in vector])
    print(vectorString, end=end)


class Ghost_Batch_Norm(
    nn.Module):  # https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition.
    def __init__(self, features, ghost_batch_size=32, number_of_ghost_batches=64, n_averaging=1, stride=1, eta=0.9,
                 bias=True, device='cpu', name='', conv=False, conv_transpose=False,
                 features_out=None):  # number_of_ghost_batches was initially set to 64
        super(Ghost_Batch_Norm, self).__init__()
        self.name = name
        self.index = None
        self.stride = stride if not conv_transpose else 1
        self.device = device
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
        m64 = x.mean(dim=0, keepdim=True, dtype=torch.float64)  # .to(self.device)

        self.m = m64.type(torch.float32).to(self.device)
        self.s = x.std(dim=0, keepdim=True).to(self.device)
        self.initialized = True
        self.running_stats = False
        self.print()

    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = torch.tensor(n_ghost_batches, dtype=torch.long).to(self.device)

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


# some basic four-vector operations
#
def PxPyPzE(v):  # need this to be able to add four-vectors
    pt = v[:, 0:1]
    eta = v[:, 1:2]
    phi = v[:, 2:3]
    m = v[:, 3:4]

    Px, Py, Pz = pt * phi.cos(), pt * phi.sin(), pt * eta.sinh()
    E = (pt ** 2 + Pz ** 2 + m ** 2).sqrt()

    return torch.cat((Px, Py, Pz, E), 1)


def PtEtaPhiM(v):
    px = v[:, 0:1]
    py = v[:, 1:2]
    pz = v[:, 2:3]
    e = v[:, 3:4]

    Pt = (px ** 2 + py ** 2).sqrt()
    ysign = 1 - 2 * (
                py < 0).float()  # if py==0, px==Pt and acos(1)=pi/2 so we need zero protection on py.sign() --> changed to the current shape to avoid 0-gradient of .sign()
    Phi = (px / Pt).acos() * ysign
    Eta = (pz / Pt).asinh()

    M = F.relu(e ** 2 - px ** 2 - py ** 2 - pz ** 2).sqrt()

    return torch.cat((Pt, Eta, Phi, M), 1)


def addFourVectors(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None):  # output added four-vectors
    # vX[batch index, (pt,eta,phi,m), object index]

    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)

    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    v12 = PtEtaPhiM(v12PxPyPzE)

    return v12, v12PxPyPzE


def calcDeltaPhi(v1, v2):  # expects eta, phi representation
    dPhi12 = (v1[:, 2:3] - v2[:, 2:3]) % math.tau
    dPhi21 = (v2[:, 2:3] - v1[:, 2:3]) % math.tau
    dPhi = torch.min(dPhi12, dPhi21)
    return dPhi


def setLeadingEtaPositive(batched_v) -> torch.Tensor:  # expects [batch, feature, jet nb]
    etaSign = 1 - 2 * (batched_v[:, 1, 0:1] < 0).float()  # -1 if eta is negative, +1 if eta is zero or positive
    batched_v[:, 1, :] = etaSign * batched_v[:, 1, :]
    return batched_v


def setLeadingPhiTo0(batched_v) -> torch.Tensor:  # expects [batch, feature, jet nb]
    # set phi = 0 for the leading jet and rotate the event accordingly
    phi_ref = batched_v[:, 2, 0:1]  # get the leading jet phi for each event
    batched_v[:, 2, :] = batched_v[:, 2, :] - phi_ref  # rotate all phi to make phi_lead = 0
    batched_v[:, 2, :][batched_v[:, 2, :] > torch.pi] -= 2 * torch.pi  # retransform the phi that are > pi
    batched_v[:, 2, :][batched_v[:, 2, :] < -torch.pi] += 2 * torch.pi  # same for the phi that are < -pi
    return batched_v


def setSubleadingPhiPositive(batched_v) -> torch.Tensor:  # expects [batch, feature, jet nb]
    batched_v[:, 2, 1:4][batched_v[:, 2, 1] < 0] += torch.pi  # starting from phi2, add pi to j2, j3, j4 if phi2 < 0
    batched_v[:, 2, :][batched_v[:, 2, :] > torch.pi] -= 2 * torch.pi  # retransform the phi that are > pi

    # phiSign = 1-2*(batched_v[:,2,1:2]<0).float() # -1 if phi2 is negative, +1 if phi2 is zero or positive
    # batched_v[:,2,1:4] = phiSign * batched_v[:,2,1:4]
    return batched_v


#
# Some different non-linear units
#
def SiLU(x):  # SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)


def NonLU(x):  # Pick the default non-Linear Unit
    return SiLU(x)  # often slightly better performance than standard ReLU
    # return F.relu(x)
    # return F.rrelu(x, training=training)
    # return F.leaky_relu(x, negative_slope=0.1)
    # return F.elu(x)


#
# embed inputs in feature space
#
class Simple_Input_Embed(nn.Module):
    def __init__(self, dimension, device='cpu'):
        super(Simple_Input_Embed, self).__init__()
        self.d = dimension
        self.device = device

        # embed inputs to dijetResNetBlock in target feature space
        self.embed = Ghost_Batch_Norm(3, features_out=self.d, conv=True, name='jet embedder',
                                      device=self.device)  # phi is relative to dijet, mass is zero in toy data. # 3 features -> 8 features
        self.conv = Ghost_Batch_Norm(self.d, conv=True, name='jet convolution', device=self.device)

        # self.register_buffer('tau', torch.tensor(math.tau, dtype=torch.float))

    def data_prep(self, j):
        j = j.clone()  # prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        j = j.view(-1, 4, 4)

        # set up all possible jet pairings
        j = torch.cat([j, j[:, :, (0, 2, 1, 3)], j[:, :, (0, 3, 1, 2)]], 2)

        # take log of pt, mass variables which have long tails
        j[:, (0, 3), :] = torch.log(1 + j[:, (0, 3), :])

        return j

    def set_mean_std(self, j):
        j = self.data_prep(j)
        self.embed.set_mean_std(j[:, 0:3])  # mass is always zero in toy data

    def set_ghost_batches(self, n_ghost_batches):
        self.embed.set_ghost_batches(n_ghost_batches)
        self.conv.set_ghost_batches(n_ghost_batches)

    def forward(self, j):
        j = self.data_prep(j)
        j = self.embed(j[:, 0:3])
        j = self.conv(NonLU(j))
        return j


class New_AE(nn.Module):
    def __init__(self, dimension, bottleneck_dim=None, out_features=12, device='cpu'):
        super(New_AE, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.out_features = out_features
        self.n_ghost_batches = 64

        self.name = f'New_AE_{self.d}'

        self.input_embed = Ghost_Batch_Norm(3, features_out=self.d, conv=True, name='jet input embedder',
                                            device=self.device)
        # mass is 0 in toy data
        self.encoder_conv = Ghost_Batch_Norm(self.d, conv=True, name='jet encoder convolution', device=self.device)

        self.bottleneck_in = Ghost_Batch_Norm(self.d, features_out=self.d_bottleneck, conv=True, device=self.device)
        self.bottleneck_out = Ghost_Batch_Norm(self.d_bottleneck, features_out=self.d, conv=True, device=self.device)

        self.decoder_conv = Ghost_Batch_Norm(self.d, conv=True, name='jet encoder convolution', device=self.device)
        self.output_recon = Ghost_Batch_Norm(self.d, features_out=3, conv=True, name='jet input embedder',
                                             device=self.device)

    def data_prep(self, j):
        j = j.clone()  # prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        j = j.view(-1, 4, 4)

        # take log of pt, mass variables which have long tails
        j[:, (0, 3), :] = torch.log(1 + j[:, (0, 3), :])

        # set up all possible jet pairings
        j = torch.cat([j, j[:, :, (0, 2, 1, 3)], j[:, :, (0, 3, 1, 2)]], 2)

        return j

    def set_mean_std(self, j):
        prep = self.data_prep(j)
        self.input_embed.set_mean_std(prep[:, 0:3])  # mass is always zero in toy data

    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = n_ghost_batches
        # encoder
        self.input_embed.set_ghost_batches(n_ghost_batches)
        self.encoder_conv.set_ghost_batches(n_ghost_batches)
        # bottleneck
        self.bottleneck_in.set_ghost_batches(n_ghost_batches)
        self.bottleneck_out.set_ghost_batches(n_ghost_batches)
        # decoder
        self.decoder_conv.set_ghost_batches(n_ghost_batches)
        self.output_recon.set_ghost_batches(n_ghost_batches)

    def forward(self, j):
        j_rot = j.clone()

        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane

        # maybe rotate the jets at the end, or take it out. Also it could be possible to rotate jets at the end to match the initial jets
        j_rot = setSubleadingPhiPositive(
            setLeadingPhiTo0(setLeadingEtaPositive(j_rot))) if self.phi_rotations else j_rot

        # convert to PxPyPzE and compute means and variances
        jPxPyPzE = PxPyPzE(j_rot)

        # j_rot.shape = [batch_size, 4, 4]
        #
        # Encode Block
        #
        j = self.data_prep(j_rot)
        j = NonLU(self.input_embed(j))
        j = NonLU(self.encoder_conv(j))
        j = NonLU(self.bottleneck_in(j))

        #
        # Decode Block
        #
        j = NonLU(self.bottleneck_out(j))
        j = NonLU(self.decoder_conv(j))
        j = NonLU(self.output_recon(j))


        Pt = j[:, 0:1].cosh() + 39  # ensures pt is >=40 GeV
        Px = Pt * j[:, 2:3].cos()
        Py = Pt * j[:, 2:3].sin()
        Pz = Pt * j[:, 1:2].sinh()
        # M  =    dec_j[:,3:4].cosh()-1 # >=0, in our case it is always zero for the toy data. we could relax this for real data
        # E  = (Pt**2+Pz**2+M**2).sqrt()   # ensures E^2>=M^2
        E = (Pt ** 2 + Pz ** 2).sqrt()  # ensures E^2>=0. In our case M is zero so let's not include it
        rec_jPxPyPzE = torch.cat((Px, Py, Pz, E), 1)

        return jPxPyPzE, rec_jPxPyPzE


