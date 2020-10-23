#!/usr/bin/python3
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


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


def bdspl(x, t, k):
    n = len(t) - k - 1
    d = np.asarray([k * (B(x, k-1, i, t) / (t[i + k] - t[i]) - B(x, k-1, i+1, t) / (t[i+k+1] - t[i+1])) for i in range(n)])
    d = np.nan_to_num(d, nan=0.)
    return d


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
    if verbosity > 2:
        plt.ion()
        fig = plt.gcf()
        b_line_a, = plt.plot(x, bspline_pol(x)[:, 0], label='B Spline A')
        real_a, = plt.plot(x, y_pol[:, 0], label='Real function A')
        b_line_b, = plt.plot(x, bspline_cpd(x)[:, 0], label='B Spline B')
        real_b, = plt.plot(x, y_cpd[:, 0], label='Real function B')
        plt.legend(loc='upper right')
        fig.canvas.draw()
        fig.canvas.flush_events()

    while True:
        if verbosity > 2:
            b_line_a.set_ydata(bspline_pol(x)[:, 0])
            real_a.set_ydata(y_pol[:, 0])
            b_line_b.set_ydata(bspline_cpd(x)[:, 0])
            real_b.set_ydata(y_cpd[:, 0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        result_ode_pol = bd.dot(ode_pol.calc(u_pol, u_cpd))
        result_ode_cpd = bd.dot(ode_cpd.calc(u_pol, u_cpd))
        b_coff_pol, _, _, _ = np.linalg.lstsq(b_lhs, spl.dot(y_pol) + w_model * result_ode_pol)
        b_coff_cpd, _, _, _ = np.linalg.lstsq(b_lhs, spl.dot(y_cpd) + w_model * result_ode_cpd)

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
        ode_pol.coeff = a_coff_pol
        ode_cpd.coeff = a_coff_cpd
        bspline_pol.c = b_coff_pol
        bspline_cpd.c = b_coff_cpd
        u_pol = bspline_pol(x)
        u_cpd = bspline_cpd(x)

    if verbosity > 2:
        plt.ioff()
        plt.close('all')

    return ode_pol, ode_cpd
