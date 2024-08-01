import torch
import torch.nn as nn

from .base import PreprocessBase
from ... import utils
class PreprocessOriginal(PreprocessBase):
    def __init__(self, phi_rotations=False, return_masses=False):
        super().__init__()
        self.phi_rotations = phi_rotations
        self.return_masses = return_masses

    def forward(self, j):
        j_rot = j.clone()  # j.shape = [batch_size, 4, 4]

        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane
        if self.phi_rotations:
            j_rot = utils.setSubleadingPhiPositive(
                utils.setLeadingPhiTo0(utils.setLeadingEtaPositive(j_rot)))

        # convert to PxPyPzE and compute means and variances
        jPxPyPzE = utils.PxPyPzE(j_rot)  # j_rot.shape = [batch_size, 4, 4]

        j_rot = j_rot.view(-1, 4, 4)

        if self.return_masses:
            d, dPxPyPzE = utils.addFourVectors(j_rot[:, :, (0, 2, 0, 1, 0, 1)],  # 6 pixels
                                               j_rot[:, :, (1, 3, 2, 3, 3, 2)])

            q, _ = utils.addFourVectors(d[:, :, (0, 2, 4)],
                                        d[:, :, (1, 3, 5)],
                                        v1PxPyPzE=dPxPyPzE[:, :, (0, 2, 4)],
                                        v2PxPyPzE=dPxPyPzE[:, :, (1, 3, 5)])
            m2j = d[:, 3:4, :].clone()
            m4j = q[:, 3:4, :].clone()

        # take log of pt, mass variables which have long tails
        j_rot[:, (0, 3), :] = torch.log(1 + j_rot[:, (0, 3), :])

        if self.return_masses:
            return j_rot, m2j, m4j
        else:
            return j_rot
