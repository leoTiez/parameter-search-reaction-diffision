#!/usr/bin/python3
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

import seqDataHandler as dh

VERBOSITY = 1


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
    def __init__(self, coeff, restrictions, num_sys=10000):
        self._coeff = np.array([coeff, ] * num_sys)
        self.restrictions = restrictions
        self.num_sys = num_sys

    def calc(self, diffs_pol, diffs_cpd, dt=1.):
        conc = self.conc_params(diffs_pol, diffs_cpd)
        return dt * np.einsum('ki,ijk->jk', self._coeff, conc)

    def conc_params(self, diffs_pol, diffs_cpd):
        spatial_diff = nabla_sq_1d(diffs_pol)
        conc = np.asarray([diffs_pol, diffs_cpd, spatial_diff])
        return conc[~self.restrictions, :, :]

    def set_coff(self, new_coffs):
        self._coeff = new_coffs


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


def fit(
        y_pol,
        y_cpd,
        ode_pol,
        ode_cpd,
        x=np.asarray([0, 3, 6, 9]),
        degree=3,
        delta=1e-8,
        success_ratio=0.97,
        verbosity=0
):
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
        conc_mat_pol = np.matmul(pol_ode_conc.transpose(2, 0, 1), pol_ode_conc.transpose(2, 1, 0))
        conc_mat_cpd = np.matmul(cpd_ode_conc.transpose(2, 0, 1), cpd_ode_conc.transpose(2, 1, 0))
        r_term_pol = np.einsum('ij,ijk->ik', bspline_pol.c.T.dot(bd.T), pol_ode_conc.T)
        r_term_cpd = np.einsum('ij,ijk->ik', bspline_cpd.c.T.dot(bd.T), cpd_ode_conc.T)

        a_coff_pol = []
        a_coff_cpd = []
        for c_pol, c_cpd, r_pol, r_cpd in zip(conc_mat_pol, conc_mat_cpd, r_term_pol, r_term_cpd):
            ac_pol, _, _, _ = np.linalg.lstsq(c_pol, 2 * r_pol)
            ac_cpd, _, _, _ = np.linalg.lstsq(c_cpd, 2 * r_cpd)
            a_coff_pol.append(ac_pol)
            a_coff_cpd.append(ac_cpd)

        a_coff_pol = np.asarray(a_coff_pol)
        a_coff_cpd = np.asarray(a_coff_cpd)

        success_pol = (
                              np.linalg.norm(b_coff_pol - bspline_pol.c, axis=0) < delta
                      ).astype('int').sum() / float(y_pol.shape[1])

        success_cpd = (
                              np.linalg.norm(b_coff_cpd - bspline_cpd.c, axis=0) < delta
                      ).astype('int').sum() / float(y_cpd.shape[1])
        if success_pol > success_ratio and success_cpd > success_ratio:
            break

        if verbosity > 0:
            print('MAX Diff b spline factors pol %s' % np.linalg.norm(b_coff_pol - bspline_pol.c, axis=0).max())
            print('MIN Diff b spline factors pol %s' % np.linalg.norm(b_coff_pol - bspline_pol.c, axis=0).min())
            print('RATIO b spline factors pol under thresh %s' % success_pol)
            print('MAX Diff b spline factors cpd %s' % np.linalg.norm(b_coff_cpd - bspline_cpd.c, axis=0).max())
            print('MIN Diff b spline factors cpd %s' % np.linalg.norm(b_coff_cpd - bspline_cpd.c, axis=0).min())
            print('RATIO b spline factors cpd under thresh %s\n' % success_cpd)

        ode_pol.set_coff(a_coff_pol)
        ode_cpd.set_coff(a_coff_cpd)
        bspline_pol.c = b_coff_pol
        bspline_cpd.c = b_coff_cpd
        u_pol = bspline_pol(x)
        u_cpd = bspline_cpd(x)

    return ode_pol, ode_cpd


def gravity_centre(hist, bins, ratio=0.8):
    s = 0
    s_total = hist.sum()
    h, b = [], []
    while True:
        s += hist.max()
        h.append(hist.max())
        b.append(bins[hist == hist.max()][0])
        bins = bins[hist < hist.max()]
        hist = hist[hist < hist.max()]
        if s > ratio * s_total:
            break
    b = np.asarray(b).reshape(-1)
    h = np.asarray(h).reshape(-1)

    return b, h


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
    equilibrium = np.asarray(all_values[4])[int(1e6):int(1.01e6)]
    pol_t0 = np.asarray(all_values[0])[int(1e6):int(1.01e6)] - equilibrium
    pol_t30 = np.asarray(all_values[2])[int(1e6):int(1.01e6)] - equilibrium
    cpd_t0 = np.asarray(all_values[1])[int(1e6):int(1.01e6)]
    cpd_t30 = np.asarray(all_values[3])[int(1e6):int(1.01e6)]

    del all_values

    if VERBOSITY > 1:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))

        ax[0].hist(pol_t0, bins='auto', alpha=0.4, label='t=0')
        ax[0].hist(pol_t30, bins='auto', alpha=0.4, label='t=30')
        ax[0].set_title('Histogram for Pol II')
        ax[0].legend()

        ax[1].hist(pol_t0[np.abs(pol_t0) > 1.], bins='auto', alpha=0.4, label='t=0 non-zero')
        ax[1].hist(pol_t30[np.abs(pol_t30) > 1.], bins='auto', alpha=0.4, label='t=30 non-zero')
        ax[1].set_title('Histogram for Pol II with deviation > 1')
        ax[1].legend()

        fig_c, ax_c = plt.subplots(2, 1, figsize=(12, 7))

        ax_c[0].hist(cpd_t0, bins='auto', alpha=0.4, label='t=0')
        ax_c[0].hist(cpd_t30, bins='auto', alpha=0.4, label='t=30')
        ax_c[0].set_title('Histogram for CPD')
        ax_c[0].legend()

        ax_c[1].hist(cpd_t0[np.abs(cpd_t0) > 1.], bins='auto', alpha=0.4, label='t=0 non-zero')
        ax_c[1].hist(cpd_t30[np.abs(cpd_t30) > 1.], bins='auto', alpha=0.4, label='t=30 non-zero')
        ax_c[1].set_title('Histogram for CPD with deviation > 1')
        ax_c[1].legend()

        plt.show()

    ode_pol = ODE(np.asarray([-1, 2, 0.01]), np.asarray([False, False, True]), num_sys=pol_t0.size)
    ode_cpd = ODE(np.asarray([-1, 0, 0]), np.asarray([False, True, True]), num_sys=cpd_t0.size)

    # TODO Problem when changing restrictions
    ode_pol, ode_cpd = fit(
        np.asarray([equilibrium + pol_t0, equilibrium + pol_t30, equilibrium + np.zeros(pol_t30.size), equilibrium + np.zeros(pol_t30.size)]),
        np.asarray([cpd_t0, cpd_t30, np.zeros(cpd_t30.size), np.zeros(pol_t30.size)]),
        ode_pol,
        ode_cpd,
        degree=3,
        verbosity=VERBOSITY
    )

    if VERBOSITY > 0:
        print('Pol coefficients have a mean of %s with a variance of %s and a std of %s' %
              (ode_pol._coeff.mean(axis=0), ode_pol._coeff.var(axis=0), ode_pol._coeff.std(axis=0)))

        print('CPD coefficients have a mean of %s with a variance of %s and a std of %s' %
              (ode_cpd._coeff.mean(axis=0), ode_cpd._coeff.var(axis=0), ode_cpd._coeff.std(axis=0)))

    pol_param = np.zeros(3)
    for c in range(ode_pol._coeff.shape[1]):
        hist, bins = np.histogram(ode_pol._coeff[:, c], bins='auto')
        bins = bins[:-1]
        b, h = gravity_centre(hist, bins)
        pol_param[c] = b.mean()

        if VERBOSITY > 1:
            plt.bar(b, h, width=0.5, alpha=0.4, label='Coeff %s' % c)

    if VERBOSITY > 1:
        plt.legend()
        plt.show()

    cpd_param = np.zeros(3)
    for c in range(ode_cpd._coeff.shape[1]):
        hist, bins = np.histogram(ode_cpd._coeff[:, c], bins='auto')
        bins = bins[:-1]
        b, h = gravity_centre(hist, bins)
        cpd_param[c] = b.mean()
        if VERBOSITY > 1:
            plt.bar(b, h, width=0.5, alpha=0.4, label='Coeff %s' % c)

    if VERBOSITY > 1:
        plt.legend()
        plt.show()

    # TODO Why is the spatial influence a problem?
    pol_param[2] = 0
    ode_pol = ODE(pol_param, np.full(3, False), num_sys=pol_t0.size)
    ode_cpd = ODE(cpd_param, np.asarray([False, True, True]), num_sys=cpd_t0.size)
    pos = np.arange(0, pol_t0.size)

    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    line_pol, = ax.plot(pos, equilibrium + pol_t0, label='Pol2')
    line_cpd, = ax.plot(pos, cpd_t0, label='CPD')
    ax.plot(pos, equilibrium + pol_t30, label='Pol2 30')
    ax.plot(pos, cpd_t30, label='CPD 30')
    ax.plot(pos, equilibrium + pol_t0, label='Pol2 0')
    ax.plot(pos, cpd_t30, label='CPD 0')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    fig.canvas.flush_events()

    pol = (equilibrium + pol_t0).reshape(1, pol_t0.size)
    cpd = cpd_t0.reshape(1, cpd_t0.size)
    for time in range(10000):
        pol_new = ode_pol.calc(pol, cpd, dt=0.01)
        cpd_new = ode_cpd.calc(pol, cpd, dt=0.01)
        cpd += cpd_new
        pol += pol_new

        line_pol.set_ydata(pol.reshape(-1))
        line_cpd.set_ydata(cpd.reshape(-1))
        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == '__main__':
    main()

