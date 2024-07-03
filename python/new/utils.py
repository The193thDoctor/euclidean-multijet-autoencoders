import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def vector_print(vector, end='\n'):
    vectorString = ", ".join([f'{element:7.2f}' for element in vector])
    print(vectorString, end=end)


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
        self.updates = 0

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
        remaining_dim = x.shape[1:]

        if self.training and self.n_ghost_batches != 0:
            # this has been changed from self.ghost_batch_size = batch_size // self.n_ghost_batches.abs()
            self.ghost_batch_size = torch.div(batch_size, self.n_ghost_batches.abs(), rounding_mode='trunc')

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = x.view(self.n_ghost_batches.abs(), self.ghost_batch_size, *remaining_dim)
    

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
                self.m = self.eta * self.m + (self.one - self.eta) * bm
                self.s = self.eta * self.s + (self.one - self.eta) * bs
            else:
                self.m = bm
                self.s = bs

            x = (x - gbm) / gbs

        else:
            x = (x - self.m) / self.s

        x = x * self.gamma + self.bias
        # back to standard indexing for convolutions: [batch, feature, pixel]
        x = x.view(batch_size, *remaining_dim)
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


def calcDeltaEta(v1, v2):  # expects PtEtaPhiM representation
    dEta = (v2[:, 1:2, :] - v1[:, 1:2, :])
    return dEta


def calcDeltaPhi(v1, v2):  # expects eta, phi representation
    dPhi12 = (v1[:, 2:3] - v2[:, 2:3]) % math.tau
    dPhi21 = (v2[:, 2:3] - v1[:, 2:3]) % math.tau
    dPhi = torch.min(dPhi12, dPhi21)
    return dPhi


def calcDeltaR(v1, v2):  # expects PtEtaPhiM representation
    dEta = (v1[:, 1:2, :] - v2[:, 1:2, :])
    dPhi = calcDeltaPhi(v1, v2)
    return (dEta ** 2 + dPhi ** 2).sqrt()


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


def deltaR_correction(dec_j) -> torch.Tensor:
    deltaPhi = calcDeltaPhi(dec_j[:, :, (0, 2, 0, 1, 0, 1)], dec_j[:, :, (1, 3, 2, 3, 3, 2)])
    deltaEta = calcDeltaEta(dec_j[:, :, (0, 2, 0, 1, 0, 1)], dec_j[:, :, (1, 3, 2, 3, 3, 2)])
    inputDeltaR_squared = deltaEta ** 2 + deltaPhi ** 2  # get the DeltaR squared between jets
    closest_pair_of_dijets_deltaR, closest_pair_of_dijets_indices = torch.topk(inputDeltaR_squared[:, 0, :], k=2,
                                                                               largest=False)  # get the 2 minimum DeltaR and their indices
    # here I use [:,0,:] because second dimension is feature, which can be suppressed once youre
    # only dealing with DeltaR
    if torch.any(closest_pair_of_dijets_deltaR < 0.16):  # if any of them has a squared deltaR < 0.16
        less_than_016 = torch.nonzero(
            torch.lt(closest_pair_of_dijets_deltaR, 0.16))  # gives the [event_idx, pairing_idx_idx]
        # event_idx gives the number of the event in which theres a pairing with DeltaR < 0.4
        # pairing_idx_idx gives 0 or 1, depending if the pairing with DeltaR < 0.4 is the smallest or the second smallest, respectively
        # imagine that in the same event (call it number 65), both 01 and 23 dijets have DeltaR < 0.4, then you will get 65 twice in event_idx
        # like this [..., [69, 0], [69, 1], ...]
        # this 0 or 1 should be indexed in closest_pair_of_dijets_indices, which will give you the true pairing index (i.e. 0-5)
        event_idx = less_than_016[:, 0]
        pairing_idx_idx = less_than_016[:, 1]

        # now we get the true index 0-5 of the pairing (0: 01, 1: 23, 2: 02, 3: 13, 4: 03, 5: 12)
        pairing_idx = closest_pair_of_dijets_indices[event_idx, pairing_idx_idx]

        # clone the tensor to be modified keeping only the events that will be corrected
        dec_j_temp = dec_j[event_idx].clone()

        # get a [nb_events_to_be_modified, 1, 4] tensor, in which each of the elements of the list corresponds to jet's individual eta and phi that shall be corrected
        first_elements_to_modify = dec_j_temp[:, :, (0, 2, 0, 1, 0, 1)][torch.arange(dec_j_temp.shape[0]), :,
                                   pairing_idx].unsqueeze(1)[:, :, 1:3]
        second_elements_to_modify = dec_j_temp[:, :, (1, 3, 2, 3, 3, 2)][torch.arange(dec_j_temp.shape[0]), :,
                                    pairing_idx].unsqueeze(1)[:, :, 1:3]

        # get a [nb_events_to_be_modified, 1, 1] tensor that parametrizes the movement along the line that joins the two points (eta1, phi1) - (eta2, phi2)
        t = (0.4 * (1 + inputDeltaR_squared[event_idx, :, pairing_idx].sqrt()) / inputDeltaR_squared[event_idx, :,
                                                                                 pairing_idx].sqrt()).unsqueeze(1)

        # this will displace the second elements along the line that joins them so that their separation is 0.16
        # modify eta
        second_elements_to_modify[:, :, 0:1] = first_elements_to_modify[:, :, 0:1] + t * (
                    second_elements_to_modify[:, :, 0:1] - first_elements_to_modify[:, :, 0:1])

        # modify phi
        phi_diff = torch.min((second_elements_to_modify[:, :, 1:2] - first_elements_to_modify[:, :, 1:2]) % math.tau,
                             (first_elements_to_modify[:, :, 1:2] - second_elements_to_modify[:, :, 1:2]) % math.tau)
        # if modifying phi2, the slope is marked by the sign from phi2 to phi1
        phi_slope_sign = torch.sign(second_elements_to_modify[:, :, 1:2] - first_elements_to_modify[:, :, 1:2])
        second_elements_to_modify[:, :, 1:2] = first_elements_to_modify[:, :, 1:2] + t * phi_slope_sign * phi_diff

        # express all pi in the range (-pi, pi)
        second_elements_to_modify[second_elements_to_modify[:, 0, 1] < -torch.pi, :, 1] += 2 * torch.pi
        second_elements_to_modify[second_elements_to_modify[:, 0, 1] > +torch.pi, :, 1] -= 2 * torch.pi

        second_pairing_idx = torch.tensor([1, 3, 2, 3, 3, 2], dtype=torch.int32)
        jet_idx = second_pairing_idx[pairing_idx]

        for i, idx in enumerate(event_idx):
            dec_j[idx, 1:3, jet_idx[i]] = second_elements_to_modify[i].squeeze()
    else:
        pass
    return dec_j


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