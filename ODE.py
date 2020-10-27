#!/usr/bin/python3
import numpy as np


def nabla_sq_1d(substance, h=1, axis=1):
    substance_xnh = np.roll(substance, -h, axis=axis)
    substance_xph = np.roll(substance, h, axis=axis)
    return substance_xph + substance_xnh - 2 * substance


def move_forward(substance, h=1, axis=1):
    return np.roll(substance, h, axis=axis)


class ODE:
    own_idx = 0

    def __init__(self, coeff, restrictions, spatial_d=nabla_sq_1d, num_sys=10000, auto_pow=None):
        self.restrictions = restrictions
        self.spatial_d = spatial_d
        self.num_sys = num_sys
        self.auto_pow = auto_pow

        self.coeff = None
        self.set_uniform_coeff(coeff)

        self.own_idx = ODE.own_idx
        ODE.own_idx += 1

    def calc(self, diffs, dt=.1):
        conc = self.conc_params(diffs)
        return dt * np.einsum('ki,ijk->jk', self.coeff, conc)

    def conc_params(self, diffs):
        own_spec = diffs[self.own_idx]
        spatial_diff = self.spatial_d(own_spec) if self.spatial_d is not None else np.zeros(own_spec.shape)
        if self.auto_pow is not None:
            conc = np.dstack((diffs.T, np.power(own_spec, self.auto_pow).T, spatial_diff.T)).T
        else:
            conc = np.dstack((diffs.T, spatial_diff.T)).T
        return conc[~self.restrictions, :, :]

    def set_uniform_coeff(self, coeff):
        self.coeff = np.array([coeff[~self.restrictions], ] * self.num_sys)

