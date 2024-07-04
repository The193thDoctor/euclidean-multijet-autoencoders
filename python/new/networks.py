import torch
import torch.nn as nn
from new import utils

torch.manual_seed(0)  # make training results repeatable

class New_AE(nn.Module):
    def __init__(self, dimension, bottleneck_dim=None, permute_input_jet=False, phi_rotations=False,
                 correct_DeltaR=False, return_masses=False):
        super(New_AE, self).__init__()
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.n_ghost_batches = 64
        self.permute_input_jet = permute_input_jet
        self.phi_rotations = phi_rotations
        self.correct_DeltaR = correct_DeltaR
        self.return_masses = return_masses

        self.name = f'New_AE_{self.d}'




        self.input_embed = nn.Conv1d(3, self.d, 1)
        self.input_embed_bn = utils.Ghost_Batch_Norm(self.d)
        # mass is 0 in toy data
        self.encoder_conv = nn.Conv1d(self.d, self.d, 1)
        self.encoder_conv_bn = utils.Ghost_Batch_Norm(self.d)

        self.bottleneck_in = nn.Conv1d(self.d, self.d_bottleneck,1)
        self.bottleneck_in_bn = utils.Ghost_Batch_Norm(self.d_bottleneck)
        self.bottleneck_out = nn.Conv1d(self.d_bottleneck, self.d, 1)
        self.bottleneck_out_bn = utils.Ghost_Batch_Norm(self.d)

        self.decoder_conv = nn.Conv1d(self.d, self.d, 1)
        self.decoder_conv_bn = utils.Ghost_Batch_Norm(self.d)
        self.output_recon = nn.Conv1d(self.d,3, 1)
        self.output_recon_bn = utils.Ghost_Batch_Norm(3)

    def data_prep(self, j):
        j = j.clone()  # prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        j = j.view(-1, 4, 4)

        if self.return_masses:
            d, dPxPyPzE = utils.addFourVectors(j[:, :, (0, 2, 0, 1, 0, 1)],  # 6 pixels
                                         j[:, :, (1, 3, 2, 3, 3, 2)])

            q, _ = utils.addFourVectors(d[:, :, (0, 2, 4)],
                                         d[:, :, (1, 3, 5)],
                                         v1PxPyPzE=dPxPyPzE[:, :, (0, 2, 4)],
                                         v2PxPyPzE=dPxPyPzE[:, :, (1, 3, 5)])
            m2j = d[:, 3:4, :].clone()
            m4j = q[:, 3:4, :].clone()

        # take log of pt, mass variables which have long tails
        j[:, (0, 3), :] = torch.log(1 + j[:, (0, 3), :])

        # set up all possible jet pairings
        # j = torch.cat([j, j[:, :, (0, 2, 1, 3)], j[:, :, (0, 3, 1, 2)]], 2)

        if self.return_masses:
            return j, m2j, m4j
        else:
            return j

    def set_mean_std(self, j):
        return

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
        #
        # Preparation block
        #
        j_rot = j.clone()  # j.shape = [batch_size, 4, 4]

        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane
        j_rot = utils.setSubleadingPhiPositive(
            utils.setLeadingPhiTo0(utils.setLeadingEtaPositive(j_rot))) if self.phi_rotations else j_rot

        if self.permute_input_jet:
            for i in range(
                    j.shape[0]):  # randomly permute the input jets positions# randomly permute the input jets positions
                j_rot[i] = j[i, :, torch.randperm(4)]

        # convert to PxPyPzE and compute means and variances
        jPxPyPzE = utils.PxPyPzE(j_rot)  # j_rot.shape = [batch_size, 4, 4]

        # j_rot.shape = [batch_size, 4, 4]
        #
        # Encode Block
        #
        if self.return_masses:
            j, m2j, m4j = self.data_prep(j_rot)
        else:
            j = self.data_prep(j_rot)

        j = utils.NonLU(self.input_embed_bn(self.input_embed(j[:, 0:3])))
        j = j + utils.NonLU(self.encoder_conv_bn(self.encoder_conv(j)))
        j = utils.NonLU(self.bottleneck_in_bn(self.bottleneck_in(j)))

        # store latent representation
        z = j.clone()

        #
        # Decode Block
        #
        j = utils.NonLU(self.bottleneck_out_bn(self.bottleneck_out(j)))
        j = j + utils.NonLU(self.encoder_conv_bn(self.decoder_conv(j)))
        j = utils.NonLU(self.output_recon_bn(self.output_recon(j)))

        # Reconstruct output
        Pt = j[:, 0:1].cosh() + 39  # ensures pt is >=40 GeV
        Eta = j[:, 1:2]
        Phi = j[:, 2:3]
        # M  = dec_j[:,3:4].cosh()-1 # >=0, in our case it is always zero for the toy data. we could relax this for real data
        M = j[:, 3:4].cosh() - 1

        rec_j = torch.cat((Pt, Eta, Phi, M), 1)
        if self.return_masses:
            rec_d, rec_dPxPyPzE = utils.addFourVectors(rec_j[:, :, (0, 2, 0, 1, 0, 1)],
                                                 rec_j[:, :, (1, 3, 2, 3, 3, 2)])
            rec_q, rec_qPxPyPzE = utils.addFourVectors(rec_d[:, :, (0, 2, 4)],
                                                 rec_d[:, :, (1, 3, 5)])
            rec_m2j = rec_d[:, 3:4, :].clone()
            rec_m4j = rec_q[:, 3:4, :].clone()

        Px = Pt * Phi.cos()
        Py = Pt * Phi.sin()
        Pz = Pt * Eta.sinh()

        # E  = (Pt**2+Pz**2+M**2).sqrt()   # ensures E^2>=M^2
        E = (Pt ** 2 + Pz ** 2).sqrt()  # ensures E^2>=0. In our case M is zero so let's not include it

        rec_jPxPyPzE = torch.cat((Px, Py, Pz, E), 1)

        if self.return_masses:
            return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j, z, m2j, m4j, rec_m2j, rec_m4j
        else:
            return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j, z


