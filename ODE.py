#!/usr/bin/python3
import numpy as np


class ODE:
    def __init__(self, coeff, restrictions, own_idx=0, num_sys=10000):
        self.restrictions = restrictions
        self.coeff = np.array([coeff[~self.restrictions], ] * num_sys)
        self.own_idx = own_idx
        self.num_sys = num_sys

    @staticmethod
    def nabla_sq_1d(substance, h=1, axis=1):
        substance_xnh = np.roll(substance, -h, axis=axis)
        substance_xph = np.roll(substance, h, axis=axis)
        return substance_xph + substance_xnh - 2 * substance

    def calc(self, diffs_pol, diffs_cpd, dt=.1):
        conc = self.conc_params(diffs_pol, diffs_cpd)
        return dt * np.einsum('ki,ijk->jk', self.coeff, conc)

    def conc_params(self, diffs_pol, diffs_cpd):
        own_spec = [diffs_pol, diffs_cpd][self.own_idx]
        spatial_diff = self.nabla_sq_1d(own_spec)
        conc = np.asarray([diffs_pol, diffs_cpd, np.power(own_spec, 3), spatial_diff])
        return conc[~self.restrictions, :, :]
