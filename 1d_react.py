#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

import seqDataHandler as dh
from ODE import ODE
from fitting import fit
from scipy import interpolate

VERBOSITY = 3


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
    bed_file = 'data/TSS_TES_steinmetz_jacquier.mRNA.bed'
    path_pol_nouv = 'data/L3_28_UV4_Pol2_noUV.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_pol_t0 = 'data/L3_29_UV4_Pol2_T0.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_pol_t30 = 'data/L3_30_UV4_Pol2_T30.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_cpd_t0 = 'data/L3_32_UV4_CPD_T0.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'
    path_cpd_t30 = 'data/L3_33_UV4_CPD_T30.BOWTIE.SacCer3.pe.bin1.RPM.rmdup.bamCoverage.bw'

    dt = 0.1
    from_idx = int(1.001e6)
    to_idx = int(1.002e6)
    num_points = 100
    restrict_pol = np.asarray([False, False, True, True])
    restrict_cpd = np.asarray([False, True, True, True])

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

    bed = dh.load_bam_bed_file(bed_file, rel_path='')

    transcripts, _, _, _ = dh.normalise_over_annotation(bw_files, bed, normalise=False)
    equilibrium = transcripts[4][0]
    pol_t0 = transcripts[0][0] - equilibrium
    pol_t30 = transcripts[2][0] - equilibrium
    cpd_t0 = transcripts[1][0]
    cpd_t30 = transcripts[3][0]
    cpd_t30 = np.minimum(cpd_t0, cpd_t30)

    del transcripts
    # all_values, _ = dh.get_values(bw_files)
    # equilibrium = np.asarray(all_values[4])[from_idx:to_idx]
    # pol_t0 = np.asarray(all_values[0])[from_idx:to_idx] - equilibrium
    # pol_t30 = np.asarray(all_values[2])[from_idx:to_idx] - equilibrium
    # cpd_t0 = np.asarray(all_values[1])[from_idx:to_idx]
    # cpd_t30 = np.asarray(all_values[3])[from_idx:to_idx]
    # # TODO Use rescaled cpd t0 signal since basic assumption is that cpds become less and cannot re-create
    # cpd_t30 = np.minimum(cpd_t0, cpd_t30)

    # del all_values

    if VERBOSITY > 4:
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

    ode_pol = ODE(np.random.random(4), restrict_pol, num_sys=pol_t0.size)
    ode_cpd = ODE(np.random.random(4), restrict_cpd, num_sys=cpd_t0.size)

    pol_data = np.asarray([pol_t0, pol_t30, np.zeros(pol_t30.size)])
    cpd_data = np.asarray([cpd_t0, cpd_t30, np.zeros(cpd_t30.size)])

    # Interpolations
    x = np.linspace(0, 10, num_points)
    pol_inter = interpolate.interp1d(np.asarray([0, 3, 10]), pol_data.T, kind='quadratic')
    cpd_inter = interpolate.interp1d(np.asarray([0, 3, 10]), cpd_data.T, kind='quadratic')
    pol_data = pol_inter(x)
    cpd_data = cpd_inter(x)

    if VERBOSITY > 3:
        i_lin = interpolate.interp1d(np.arange(pol_data.shape[0]), pol_data.T[0], kind='linear')
        i_near = interpolate.interp1d(np.arange(pol_data.shape[0]), pol_data.T[0], kind='nearest')
        i_zero = interpolate.interp1d(np.arange(pol_data.shape[0]), pol_data.T[0], kind='zero')
        i_slin = interpolate.interp1d(np.arange(pol_data.shape[0]), pol_data.T[0], kind='slinear')
        plt.plot(x, i_lin(x), label='linear')
        plt.plot(x, i_near(x), label='nearest')
        plt.plot(x, i_zero(x), label='zero')
        plt.plot(x, i_slin(x), label='slinear')
        plt.plot(x, pol_inter(x)[0], label='quadratic')

        plt.plot(np.arange(pol_data.shape[0]), pol_data.T[0], marker='o')
        plt.legend()
        plt.show()

    ode_pol, ode_cpd = fit(
        pol_data.T,
        cpd_data.T,
        ode_pol,
        ode_cpd,
        x=x,
        degree=5,
        w_model=1.,
        success_ratio=0.9,
        verbosity=VERBOSITY
    )

    if VERBOSITY > 0:
        print('Pol coefficients have a median of %s, a mean of %s with a variance of %s and a std of %s' %
              (np.median(ode_pol.coeff, axis=0), ode_pol.coeff.mean(axis=0), ode_pol.coeff.var(axis=0), ode_pol.coeff.std(axis=0)))

        print('CPD coefficients have a median of %s, a mean of %s with a variance of %s and a std of %s' %
              (np.median(ode_cpd.coeff, axis=0), ode_cpd.coeff.mean(axis=0), ode_cpd.coeff.var(axis=0), ode_cpd.coeff.std(axis=0)))

    pol_param_ode = np.zeros(4)
    pol_param_ode[~restrict_pol] = np.median(ode_pol.coeff, axis=0)
    cpd_param_ode = np.zeros(4)
    cpd_param_ode[~restrict_cpd] = np.median(ode_cpd.coeff, axis=0)

    ode_pol = ODE(pol_param_ode, restrict_pol, num_sys=pol_t0.size)
    ode_cpd = ODE(cpd_param_ode, restrict_cpd, num_sys=cpd_t0.size)
    pos = np.arange(0, pol_t0.size)

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].plot(pos, equilibrium + pol_t0, label='Pol2 0')
    line_pol, = ax[0].plot(pos, equilibrium + pol_t0, label='Pol2')
    ax[0].plot(pos, equilibrium + pol_t30, label='Pol2 30')
    ax[0].plot(pos, equilibrium, label='Pol2 no UV')
    ax[0].legend(loc='upper right')

    ax[1].plot(pos, cpd_t0, label='CPD 0')
    line_cpd, = ax[1].plot(pos, cpd_t0, label='CPD')
    ax[1].plot(pos, cpd_t30, label='CPD 30')
    ax[1].legend(loc='upper right')

    t = plt.title('%s min' % 0)

    fig.canvas.draw()
    fig.canvas.flush_events()

    pol = pol_t0.reshape(1, pol_t0.size)
    cpd = cpd_t0.reshape(1, cpd_t0.size)
    for time in range(10000):
        t.set_text('%s min' % (10. * time * dt))
        pol_new = ode_pol.calc(pol, cpd, dt=dt)
        cpd_new = ode_cpd.calc(pol, cpd, dt=dt)
        pol += pol_new
        cpd += cpd_new

        line_pol.set_ydata(equilibrium + pol.reshape(-1))
        line_cpd.set_ydata(cpd.reshape(-1))
        fig.canvas.draw()
        fig.canvas.flush_events()


def test_main():
    # Original parameters A
    # 1, -1, -0.1, 1
    A = np.genfromtxt('data/A-rand.csv', delimiter=',')
    # Original parameters B
    # 1, -1, -0.1, 3
    B = np.genfromtxt('data/B-rand.csv', delimiter=',')
    x = np.genfromtxt('data/Time.csv', delimiter=',')

    ode_A = ODE(np.random.random(4), np.asarray([False, False, False, False]), own_idx=0, num_sys=A.shape[1])
    ode_B = ODE(np.random.random(4), np.asarray([False, False, False, False]), own_idx=1, num_sys=B.shape[1])

    ode_A, ode_B = fit(
        A[:-1],
        B[:-1],
        ode_A,
        ode_B,
        x=x,
        w_model=1.,
        degree=5, # Finding ===> Is very sensitive to the degree. Using degree of 4 or 5 works perfect
        delta=1e-10,
        verbosity=VERBOSITY,
        success_ratio=.99
    )

    if VERBOSITY > 0:
        print('A coefficients have a median of %s, mean of %s with a variance of %s and a std of %s' %
              (np.median(ode_A.coeff, axis=0), ode_A.coeff.mean(axis=0), ode_A.coeff.var(axis=0),
               ode_A.coeff.std(axis=0)))

        print('B coefficients have a median of %s, a mean of %s with a variance of %s and a std of %s' %
              (np.median(ode_B.coeff, axis=0), ode_B.coeff.mean(axis=0), ode_B.coeff.var(axis=0),
               ode_B.coeff.std(axis=0)))

    # a_param = np.zeros(4)
    # b_param = np.zeros(4)
    # a_param[:2] = np.median(ode_A.coeff, axis=0)
    # b_param[:2] = np.median(ode_B.coeff, axis=0)

    a_param = np.mean(ode_A.coeff, axis=0)
    b_param = np.mean(ode_B.coeff, axis=0)

    ode_A = ODE(a_param, np.asarray([False, False, False, False]), own_idx=0, num_sys=A.shape[1])
    ode_B = ODE(b_param, np.asarray([False, False, False, False]), own_idx=1, num_sys=B.shape[1])

    print('A params %s, B params %s' % (a_param, b_param))

    a = A[0, :].reshape(1, A.shape[1])
    b = B[0, :].reshape(1, B.shape[1])

    a_history = [a[0]]
    b_history = [b[0]]
    for _ in np.arange(3000):
        a_dev = ode_A.calc(a, b, dt=0.1)
        b_dev = ode_B.calc(a, b, dt=0.1)
        a += a_dev
        b += b_dev
        a_history.append(np.copy(a[0]))
        b_history.append(np.copy(b[0]))

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0][0].pcolor(np.asarray(A), vmin=A.min(), vmax=A.max())
    ax[0][0].set_title('Species A')
    ax[0][1].pcolor(np.asarray(B), vmin=B.min(), vmax=B.max())
    ax[0][1].set_title('Species B')
    amin, amax = np.asarray(a_history).min(), np.asarray(a_history).max()
    bmin, bmax = np.asarray(b_history).min(), np.asarray(b_history).max()
    ax[1][0].set_title('Appr. Species A')
    ax[1][0].pcolor(np.asarray(a_history), vmin=amin, vmax=amax)
    ax[1][1].set_title('Appr. Species B')
    ax[1][1].pcolor(np.asarray(b_history), vmin=bmin, vmax=bmax)

    plt.show()


if __name__ == '__main__':
    main()
    # test_main()

