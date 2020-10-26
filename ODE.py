#!/usr/bin/python3
import numpy as np


def nabla_sq_1d(substance, h=1, axis=1):
    substance_xnh = np.roll(substance, -h, axis=axis)
    substance_xph = np.roll(substance, h, axis=axis)
    return substance_xph + substance_xnh - 2 * substance


def move_forward(substance, h=1, axis=1):
    return np.roll(substance, h, axis=axis)


class ODE:
    def __init__(self, coeff, restrictions, own_idx=0, spatial_d=nabla_sq_1d, num_sys=10000):
        self.restrictions = restrictions
        self.coeff = np.array([coeff[~self.restrictions], ] * num_sys)
        self.own_idx = own_idx
        self.spatial_d = spatial_d
        self.num_sys = num_sys

    def calc(self, diffs, dt=.1):
        conc = self.conc_params(diffs)
        return dt * np.einsum('ki,ijk->jk', self.coeff, conc)

    def conc_params(self, diffs):
        own_spec = diffs[self.own_idx]
        spatial_diff = self.spatial_d(own_spec)
        conc = np.dstack((diffs.T, spatial_diff.T)).T
        return conc[~self.restrictions, :, :]
