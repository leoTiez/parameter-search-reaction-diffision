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
        y,
        odes,
        x=np.asarray([0, 3, 6, 9]),
        degree=3,
        delta=1e-8,
        w_model=1.0,
        success_ratio=0.97,
        max_iter=1200,
        verbosity=0
):
    # create example t, as they should be the same for all data values along the genome
    t, _, _ = interpolate.splrep(x, y[0][:, 0], s=0, k=degree)
    bd = bdspl(x, t, k=degree)
    spl = bspl(x, t, k=degree)

    bsplines = []
    for y_sub in y:
        bsplines.append(interpolate.make_lsq_spline(x=x, y=y_sub, t=t, k=degree))

    b_lhs = w_model * bd.dot(bd.T) + spl.dot(spl.T)
    u = []
    for bspline in bsplines:
        u.append(bspline(x))

    u = np.asarray(u)
    counter = 0
    success_old = np.full(y.shape[0], -1.)

    if verbosity > 2:
        plt.ion()
        fig = plt.gcf()
        b_spl_lines = []
        real_lines = []
        for num, (u_sub, y_sub) in enumerate(zip(u, y)):
            b_line, = plt.plot(x, u_sub[:, 0], label='B-Spline %s' % num)
            real_line, = plt.plot(x, y_sub[:, 0], label='Real function %s' % num)
            b_spl_lines.append(b_line)
            real_lines.append(real_line)

        plt.legend(loc='upper right')
        fig.canvas.draw()
        fig.canvas.flush_events()

    while True:
        if verbosity > 0:
            print('Counter %s' % counter)

        if verbosity > 2:
            for b_line, real_line, u_sub, y_sub in zip(b_spl_lines, real_lines, u, y):
                b_line.set_ydata(u_sub[:, 0])
                real_line.set_ydata(y_sub[:, 0])

            fig.canvas.draw()
            fig.canvas.flush_events()

        success = []
        alpha = []
        beta = []
        for num, (bspline, y_sub, ode) in enumerate(zip(bsplines, y, odes)):
            result_ode = bd.dot(ode.calc(u))
            b_coff, _, _, _ = np.linalg.lstsq(b_lhs, spl.dot(y_sub) + w_model * result_ode)

            ode_conc = ode.conc_params(u)
            conc_mat = np.matmul(ode_conc.transpose(2, 0, 1), ode_conc.transpose(2, 1, 0))
            r_term = np.einsum('ij,ijk->ik', bspline.c.T.dot(bd), ode_conc.T)

            a_coff = []
            for c, r in zip(conc_mat, r_term):
                ac, _, _, _ = np.linalg.lstsq(c, r)
                a_coff.append(ac)

            a_coff = np.asarray(a_coff)
            s = (
                        np.linalg.norm(b_coff - bspline.c, axis=0) < delta
                ).astype('int').sum() / float(y_sub.shape[1])
            success.append(s > success_ratio and np.abs(s - success_old[num]) < delta)
            success_old[num] = s

            alpha.append(a_coff)
            beta.append(b_coff)

            if verbosity > 0:
                print('MAX Diff b spline %s' % np.linalg.norm(b_coff - bspline.c, axis=0).max())
                print('MIN Diff b spline %s' % np.linalg.norm(b_coff - bspline.c, axis=0).min())
                print('RATIO b spline factors under thresh %s\n' % s)

        print('All Success %s' % success)
        print('Old Success%s' % success_old)
        if np.all(np.asarray(success)):
            break
        if counter > max_iter:
            break

        counter += 1

        for num, (a_coeff, b_coeff) in enumerate(zip(alpha, beta)):
            odes[num].coeff = a_coeff
            bsplines[num].c = b_coeff
            u[num] = bsplines[num](x)

    if verbosity > 2:
        plt.ioff()
        plt.close('all')

    return odes
