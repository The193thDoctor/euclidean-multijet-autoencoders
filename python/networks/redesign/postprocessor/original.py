import torch
import torch.nn as nn

from .base import BasePostprocessor
from ... import utils
class OriginalPostprocessor(BasePostprocessor):
    def __init__(self, correct_delta_r=False, return_masses=False):
        super().__init__()
        self.return_masses = return_masses
        self.correct_delta_r = correct_delta_r

    def forward(self, j):
        # apply the DeltaR correction (in inference) so that jets are separated at least deltaR = 0.4
        j = utils.deltaR_correction(j) if self.correct_delta_r and not self.training else j

        j[:, 0:1] = j[:, 0:1].cosh() + 39  # ensures pt is >=40 GeV
        j[:, 3:4] = j[:, 3:4].cosh() - 1  # ensures mass is >=0 GeV

        if self.return_masses:
            d, dPxPyPzE = utils.addFourVectors(j[:, :, (0, 2, 0, 1, 0, 1)],
                                               j[:, :, (1, 3, 2, 3, 3, 2)])
            q, _ = utils.addFourVectors(d[:, :, (0, 2, 4)], d[:, :, (1, 3, 5)],
                                        v1PxPyPzE=dPxPyPzE[:, :, (0, 2, 4)],
                                        v2PxPyPzE=dPxPyPzE[:, :, (1, 3, 5)])
            m2j = d[:, 3:4, :].clone()
            m4j = q[:, 3:4, :].clone()

        if self.return_masses:
            return j, m2j, m4j
        else:
            return j
