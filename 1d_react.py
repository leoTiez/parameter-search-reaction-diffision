#!/usr/bin/python3
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

import seqDataHandler as dh

VERBOSITY = 3


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
    def __init__(self, coeff, restrictions, own_idx=0, num_sys=10000):
        self.restrictions = restrictions
        self._coeff = np.array([coeff[~self.restrictions], ] * num_sys)
        self.own_idx = own_idx
        self.num_sys = num_sys

    def calc(self, diffs_pol, diffs_cpd, dt=0.1):
        conc = self.conc_params(diffs_pol, diffs_cpd)
        return dt * np.einsum('ki,ijk->jk', self._coeff, conc)

    def conc_params(self, diffs_pol, diffs_cpd):
        spatial_diff = nabla_sq_1d(diffs_pol)
        conc = np.asarray([diffs_pol, diffs_cpd, np.power([diffs_pol, diffs_cpd][self.own_idx], 3), spatial_diff])
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
    return np.asarray([x[i] * Bdt(x, k, i, t) for i in range(n)])


def fit(
        y_pol,
        y_cpd,
        ode_pol,
        ode_cpd,
        x=np.asarray([0, 3, 6, 9]),
        degree=3,
        delta=1e-8,
        w_model=1.0,
        success_ratio=0.97,
        verbosity=0
):
    # create example t, as they should be the same for all data values along the genome
    t, _, _ = interpolate.splrep(x, y_pol[:, 0], s=0, k=degree)
    bd = bdspl(x, t, k=degree)
    spl = bspl(x, t, k=degree)
    bspline_pol = interpolate.make_lsq_spline(x=x, y=y_pol, t=t, k=degree)
    bspline_cpd = interpolate.make_lsq_spline(x=x, y=y_cpd, t=t, k=degree)

    b_lhs = w_model * bd.dot(bd.T) + spl.dot(spl.T)
    u_pol = bspline_pol(x)
    u_cpd = bspline_cpd(x)

    counter = 0
    while True:
        if verbosity > 2:
            plt.plot(x, bspline_pol(x)[:, 0], label='B Spline A')
            plt.plot(x, y_pol[:, 0], label='Real function A')
            plt.plot(x, bspline_cpd(x)[:, 0], label='B Spline B')
            plt.plot(x, y_cpd[:, 0], label='Real function B')
            plt.legend()
            plt.show()

        result_ode_pol = bd.dot(ode_pol.calc(u_pol, u_cpd))
        result_ode_cpd = bd.dot(ode_cpd.calc(u_pol, u_cpd))
        b_coff_pol, _, _, _ = np.linalg.lstsq(b_lhs, (spl.dot(y_pol) + w_model * result_ode_pol))
        b_coff_cpd, _, _, _ = np.linalg.lstsq(b_lhs, (spl.dot(y_cpd) + w_model * result_ode_cpd))

        pol_ode_conc = ode_pol.conc_params(u_pol, u_cpd)
        cpd_ode_conc = ode_cpd.conc_params(u_pol, u_cpd)
        conc_mat_pol = np.matmul(pol_ode_conc.transpose(2, 0, 1), pol_ode_conc.transpose(2, 1, 0))
        conc_mat_cpd = np.matmul(cpd_ode_conc.transpose(2, 0, 1), cpd_ode_conc.transpose(2, 1, 0))
        r_term_pol = np.einsum('ij,ijk->ik', bspline_pol.c.T.dot(bd), pol_ode_conc.T)
        r_term_cpd = np.einsum('ij,ijk->ik', bspline_cpd.c.T.dot(bd), cpd_ode_conc.T)

        a_coff_pol = []
        a_coff_cpd = []
        for c_pol, c_cpd, r_pol, r_cpd in zip(conc_mat_pol, conc_mat_cpd, r_term_pol, r_term_cpd):
            ac_pol, _, _, _ = np.linalg.lstsq(c_pol, r_pol)
            ac_cpd, _, _, _ = np.linalg.lstsq(c_cpd, r_cpd)
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
        if counter > 1200:
            break

        if verbosity > 0:
            print('Counter %s' % counter)
            print('MAX Diff b spline factors pol %s' % np.linalg.norm(b_coff_pol - bspline_pol.c, axis=0).max())
            print('MIN Diff b spline factors pol %s' % np.linalg.norm(b_coff_pol - bspline_pol.c, axis=0).min())
            print('RATIO b spline factors pol under thresh %s' % success_pol)
            print('MAX Diff b spline factors cpd %s' % np.linalg.norm(b_coff_cpd - bspline_cpd.c, axis=0).max())
            print('MIN Diff b spline factors cpd %s' % np.linalg.norm(b_coff_cpd - bspline_cpd.c, axis=0).min())
            print('RATIO b spline factors cpd under thresh %s\n' % success_cpd)

        counter += 1
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
    dt = 0.1
    from_idx = int(1.001e6)
    to_idx = int(1.002e6)
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
    equilibrium = np.asarray(all_values[4])[from_idx:to_idx]
    pol_t0 = np.asarray(all_values[0])[from_idx:to_idx] - equilibrium
    pol_t30 = np.asarray(all_values[2])[from_idx:to_idx] - equilibrium
    cpd_t0 = np.asarray(all_values[1])[from_idx:to_idx]
    cpd_t30 = np.asarray(all_values[3])[from_idx:to_idx]
    # TODO Use rescaled cpd t0 signal since basic assumption is that cpds become less and cannot re-create
    cpd_t30 = np.minimum(cpd_t0, cpd_t30)

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

    # TODO Why is the spatial influence a problem?
    ode_pol = ODE(np.asarray([-1, 2, 0.1, 1]), np.asarray([False, False, False, True]), num_sys=pol_t0.size)
    ode_cpd = ODE(np.asarray([-1, 0, 0, 0]), np.asarray([False, True, True, True]), num_sys=cpd_t0.size)

    ode_pol, ode_cpd = fit(
        np.asarray([pol_t0, pol_t30, np.zeros(pol_t30.size)]),
        np.asarray([cpd_t0, cpd_t30, np.zeros(cpd_t30.size)]),
        ode_pol,
        ode_cpd,
        x=np.asarray([0, 3, 8]),
        degree=2,
        w_model=1.,
        verbosity=VERBOSITY
    )

    if VERBOSITY > 0:
        print('Pol coefficients have a mean of %s with a variance of %s and a std of %s' %
              (ode_pol._coeff.mean(axis=0), ode_pol._coeff.var(axis=0), ode_pol._coeff.std(axis=0)))

        print('CPD coefficients have a mean of %s with a variance of %s and a std of %s' %
              (ode_cpd._coeff.mean(axis=0), ode_cpd._coeff.var(axis=0), ode_cpd._coeff.std(axis=0)))

    pol_param = np.zeros(4)
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

    cpd_param = np.zeros(4)
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

    pol_param_ode = np.zeros(4)
    pol_param_ode[:len(pol_param)] = pol_param
    cpd_param_ode = np.zeros(4)
    cpd_param_ode[:len(cpd_param)] = cpd_param

    ode_pol = ODE(pol_param_ode, np.asarray([False, False, False, False]), num_sys=pol_t0.size)
    ode_cpd = ODE(cpd_param_ode, np.asarray([False, True, True, True]), num_sys=cpd_t0.size)
    pos = np.arange(0, pol_t0.size)

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].plot(pos, equilibrium + pol_t0, label='Pol2 0')
    line_pol, = ax[0].plot(pos, equilibrium + pol_t0, label='Pol2')
    ax[0].plot(pos, equilibrium + pol_t30, label='Pol2 30')
    ax[0].plot(pos, equilibrium, label='Pol2 no UV')

    ax[1].plot(pos, cpd_t0, label='CPD 0')
    line_cpd, = ax[1].plot(pos, cpd_t0, label='CPD')
    ax[1].plot(pos, cpd_t30, label='CPD 30')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    fig.canvas.flush_events()

    pol = pol_t0.reshape(1, pol_t0.size)
    cpd = cpd_t0.reshape(1, cpd_t0.size)
    for time in range(10000):
        pol_new = ode_pol.calc(pol, cpd, dt=dt)
        cpd_new = ode_cpd.calc(pol, cpd, dt=dt)
        cpd += cpd_new
        pol += pol_new

        line_pol.set_ydata(equilibrium + pol.reshape(-1))
        line_cpd.set_ydata(cpd.reshape(-1))
        fig.canvas.draw()
        fig.canvas.flush_events()


def test_main():
    # Original parameters A
    # 1, -1, 0.1, 1
    A = np.genfromtxt('data/A-rand-nodiff.csv', delimiter=',')
    B = np.genfromtxt('data/B-rand-nodiff.csv', delimiter=',')
    # Original parameters B
    # 1, -1, 0.1, 3
    x = np.genfromtxt('data/Time.csv', delimiter=',')

    ode_A = ODE(np.asarray([1, -1, 0.1, 1]), np.asarray([False, False, False, False]), num_sys=A.shape[1])
    ode_B = ODE(np.asarray([1, -1, 0.1, 3]), np.asarray([False, False, False, False]), num_sys=B.shape[1])

    ode_A, ode_B = fit(
        A[:-1],
        B[:-1],
        ode_A,
        ode_B,
        x=x,
        degree=3,
        verbosity=VERBOSITY,
        success_ratio=.99
    )

    if VERBOSITY > 0:
        print('A coefficients have a median of %s, mean of %s with a variance of %s and a std of %s' %
              (np.median(ode_A._coeff, axis=0), ode_A._coeff.mean(axis=0), ode_A._coeff.var(axis=0),
               ode_A._coeff.std(axis=0)))

        print('B coefficients have a median of %s, a mean of %s with a variance of %s and a std of %s' %
              (np.median(ode_B._coeff, axis=0), ode_B._coeff.mean(axis=0), ode_B._coeff.var(axis=0),
               ode_B._coeff.std(axis=0)))

    # a_param = np.zeros(4)
    # b_param = np.zeros(4)
    # a_param[:2] = np.median(ode_A._coeff, axis=0)
    # b_param[:2] = np.median(ode_B._coeff, axis=0)

    a_param = np.median(ode_A._coeff, axis=0)
    b_param = np.median(ode_B._coeff, axis=0)

    ode_A = ODE(a_param, np.asarray([False, False, False, False]), num_sys=A.shape[1])
    ode_B = ODE(b_param, np.asarray([False, False, False, False]), num_sys=B.shape[1])

    # print('A params %s, B params %s' % (a_param, b_param))

    a = A[0, :].reshape(1, A.shape[1])
    b = B[0, :].reshape(1, B.shape[1])
    a_history = []
    b_history = []
    for _ in np.arange(0, 300, 1):
        a += ode_A.calc(a, b, dt=0.1)
        b += ode_B.calc(a, b, dt=0.1)
        a_history.append(a[0])
        b_history.append(b[0])
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0][0].pcolor(np.asarray(A), vmin=A.min(), vmax=A.max())
    ax[0][1].pcolor(np.asarray(B), vmin=B.min(), vmax=B.max())
    amin, amax = np.asarray(a_history).min(), np.asarray(a_history).max()
    bmin, bmax = np.asarray(b_history).min(), np.asarray(b_history).max()
    ax[1][0].pcolor(np.asarray(a_history), vmin=amin, vmax=amax)
    ax[1][1].pcolor(np.asarray(b_history), vmin=bmin, vmax=bmax)

    plt.show()


if __name__ == '__main__':
    main()
    # test_main()

