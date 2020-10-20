#!/usr/bin/python3
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

import seqDataHandler as dh



def reaction(pol, cpd, interact_pol=0., interact_cpd=1., nonlin_break=0.1):
    return pol * interact_pol + cpd * interact_cpd - nonlin_break * pol**3


def reaction_pol(pol, cpd, interact_pol=-.1, interact_cpd=.01, nonlin_break=0., diff_coff=0.01, dt=0.1):
    return np.maximum(0, pol + dt * reaction(
        pol,
        cpd,
        interact_pol=interact_pol,
        interact_cpd=interact_cpd,
        nonlin_break=nonlin_break
    ) + diff_coff * nabla_sq_1d(pol))


def reaction_cpd(cpd, pol, interact_pol=-2, interact_cpd=0, nonlin_break=0., dt=0.1):
    return np.maximum(0, cpd + dt * reaction(
        pol,
        cpd,
        interact_pol=interact_pol,
        interact_cpd=interact_cpd,
        nonlin_break=nonlin_break
    ))


def nabla_sq_1d(substance, h=1, axis=1):
    """
    Second derivative of the vector function of the species for one dimension
    :param substance: species
    :return: second derivative of the species vector function for one dimension
    """
    substance_xnh = np.roll(substance, -h, axis=axis)
    substance_xph = np.roll(substance, h, axis=axis)
    return substance_xph + substance_xnh - 2 * substance


class ODE:
    def __init__(self, coeff, restrictions):
        self._coeff = np.zeros(coeff.size)
        self.restrictions = restrictions
        self._coeff[self.restrictions != None] = self.restrictions[self.restrictions != None]
        self.set_coff(coeff)

    def calc(self, diffs_pol, diffs_cpd):
        conc = self.conc_params(diffs_pol, diffs_cpd)
        return conc.T.dot(self._coeff).T

    def conc_params(self, diffs_pol, diffs_cpd):
        spatial_diff = nabla_sq_1d(diffs_pol)
        conc = np.asarray([diffs_pol, diffs_cpd, spatial_diff])
        return conc

    def set_coff(self, new_coffs):
        self._coeff[self.restrictions == None] = new_coffs[self.restrictions == None]


def B(x, k, i, t):
    if k == 0:
        phi_0 = np.zeros(x.shape)
        phi_0[np.logical_and(t[i] <= x, x < t[i + 1])] = 1.
        return phi_0
    if t[i+k] == t[i]:
        c1 = np.zeros(x.shape)
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = np.zeros(x.shape)
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2


def bspl(x, t, k):
    n = len(t) - k - 1
    assert (n >= k+1)
    return np.asarray([B(x, k, i, t) for i in range(n)])


def Bdt(x, k, i, t):
    if k == 1:
        phid_0 = np.zeros(x.shape)
        phid_0[np.logical_and(t[i] <= x, x < t[i + 1])] = 1.
        return phid_0
    if t[i+k] == t[i]:
        c1 = np.zeros(x.shape)
    else:
        c1 = 1./(t[i+k] - t[i]) * Bdt(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = np.zeros(x.shape)
    else:
        c2 = 1./(t[i+k+1] - t[i+1]) * Bdt(x, k-1, i+1, t)
    return k * (c1 - c2)


def bdspl(x, t, k):
    n = len(t) - k - 1
    return np.asarray([Bdt(x, k, i, t) for i in range(n)])


def fit(y_pol, y_cpd, ode_pol, ode_cpd, x=np.asarray([0, 3, 6, 9]), degree=3, delta=1e-8):
    # Bspline and recursive function differ ===> Problem with low number of data points

    # dspline = BSpline(t=t, c=np.ones(t.size), k=degree)
    # bdt_f = dspline.derivative()
    # bdt = bdt_f(t)
    # bdt = bdt.reshape((bdt.size, 1))

    # create example t, as they should be the same for all data values along the genome
    t, _, _ = interpolate.splrep(x, y_pol[:, 0], s=0, k=degree)
    bd = bdspl(x, t, k=degree)
    spl = bspl(x, t, k=degree)
    bspline_pol = interpolate.make_lsq_spline(x=x, y=y_pol, t=t, k=degree)
    bspline_cpd = interpolate.make_lsq_spline(x=x, y=y_cpd, t=t, k=degree)
    b_lhs = bd.dot(bd.T) + spl.dot(spl.T)
    u_pol = bspline_pol(x)
    u_cpd = bspline_cpd(x)

    while True:
        result_ode_pol = bd.dot(ode_pol.calc(u_pol, u_cpd))
        result_ode_cpd = bd.dot(ode_cpd.calc(u_pol, u_cpd))
        b_coff_pol, _, _, _ = np.linalg.lstsq(b_lhs, 2 * (spl.dot(y_pol) + result_ode_pol))
        b_coff_cpd, _, _, _ = np.linalg.lstsq(b_lhs, 2 * (spl.dot(y_cpd) + result_ode_cpd))
        pol_ode_conc = ode_pol.conc_params(u_pol, u_cpd)
        cpd_ode_conc = ode_cpd.conc_params(u_pol, u_cpd)
        r_term_pol = np.einsum('ij,ijk->ik', bspline_pol.c.T.dot(bd.T), pol_ode_conc.T)
        r_term_cpd = np.einsum('ij,ijk->ik', bspline_cpd.c.T.dot(bd.T), cpd_ode_conc.T)
        # does not finish. What's the problem?
        a_coff_pol, _, _, _ = np.linalg.lstsq(u_pol.T.dot(u_pol), 2 * r_term_pol)
        a_coff_cpd, _, _, _ = np.linalg.lstsq(u_cpd.T.dot(u_cpd), 2 * r_term_cpd)
        if np.linalg.norm(b_coff_pol - bspline_pol.c) < delta and np.linalg.norm(b_coff_cpd - bspline_cpd.c) < delta:
            break

        ode_pol.set_coff(a_coff_pol)
        ode_cpd.set_coff(a_coff_cpd)
        bspline_pol.c = b_coff_pol
        bspline_cpd.c = b_coff_cpd
        u_pol = bspline_pol(x)
        u_cpd = bspline_cpd(x)

    return ode_pol, ode_cpd


def main():
    path_pol_nouv = 'data/L3_28_UV4_Pol2_noUV.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_pol_t0 = 'data/L3_29_UV4_Pol2_T0.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_pol_t30 = 'data/L3_30_UV4_Pol2_T30.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_cpd_t0 = 'data/L3_32_UV4_CPD_T0.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_cpd_t30 = 'data/L3_33_UV4_CPD_T30.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    dt = 0.01
    # ###############################################################################################
    # Transcript models
    # ###############################################################################################

    bw_files = []
    for path in [path_pol_t0, path_cpd_t0, path_pol_t30, path_cpd_t30, path_pol_nouv]:
        bw_files.append(
            dh.load_big_file(
                name=path,
                rel_path=''
            )
        )

    all_values, _ = dh.get_values(bw_files)
    pol_t0 = np.asarray(all_values[0])[int(1e6):int(1.01e6)] - np.asarray(all_values[4])[int(1e6):int(1.01e6)]
    pol_t30 = np.asarray(all_values[2])[int(1e6):int(1.01e6)] - np.asarray(all_values[4])[int(1e6):int(1.01e6)]
    cpd_t0 = np.asarray(all_values[1])[int(1e6):int(1.01e6)]
    cpd_t30 = np.asarray(all_values[3])[int(1e6):int(1.01e6)]

    del all_values

    ode_pol = ODE(np.asarray([-1, 2, 0.01]), np.full(3, None))
    ode_cpd = ODE(np.asarray([-1, 0, 0]), np.asarray([None, 0, 0]))

    ode_pol, ode_cpd = fit(
        np.asarray([pol_t0, pol_t30, np.zeros(pol_t30.size), np.zeros(pol_t30.size)]),
        np.asarray([cpd_t0, cpd_t30, np.zeros(cpd_t30.size), np.zeros(pol_t30.size)]),
        ode_pol,
        ode_cpd,
        degree=3
    )

    # x = np.arange(0, pol.size)
    #
    # plt.ion()
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111)
    # line_pol, = ax.plot(x, pol, label='Pol2')
    # line_cpd, = ax.plot(x, cpd, label='CPD')
    # ax.plot(x, all_values[2][int(1e6):int(1.01e6)], label='Pol2 30')
    # ax.plot(x, all_values[3][int(1e6):int(1.01e6)], label='CPD 30')
    # ax.plot(x, all_values[0][int(1e6):int(1.01e6)], label='Pol2 0')
    # ax.plot(x, all_values[1][int(1e6):int(1.01e6)], label='CPD 0')
    # plt.legend(loc='upper right')
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    #
    # for time in range(10000):
    #     pol_new = reaction_pol(pol, cpd, dt=dt)
    #     cpd_new = reaction_cpd(cpd, pol, dt=dt)
    #     cpd = cpd_new
    #     pol = pol_new
    #
    #     line_pol.set_ydata(pol)
    #     line_cpd.set_ydata(cpd)
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()


if __name__ == '__main__':
    main()

