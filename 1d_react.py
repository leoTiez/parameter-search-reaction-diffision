#!/usr/bin/python3
import numpy as np
import seqDataHandler as dh
import matplotlib.pyplot as plt


def nabla_sq_1d(substance):
    """
    Second derivative of the vector function of the species for one dimension
    :param substance: species
    :return: second derivative of the species vector function for one dimension
    """
    substance_xn1 = np.roll(substance, -1)
    substance_xp1 = np.roll(substance, 1)
    return substance_xp1 + substance_xn1 - 2 * substance


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
    pol = np.asarray(all_values[0])[int(1e6):int(1.01e6)] - np.asarray(all_values[4])[int(1e6):int(1.01e6)]
    cpd = np.asarray(all_values[1])[int(1e6):int(1.01e6)]
    x = np.arange(0, pol.size)

    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    line_pol, = ax.plot(x, pol, label='Pol2')
    line_cpd, = ax.plot(x, cpd, label='CPD')
    ax.plot(x, all_values[2][int(1e6):int(1.01e6)], label='Pol2 30')
    ax.plot(x, all_values[3][int(1e6):int(1.01e6)], label='CPD 30')
    ax.plot(x, all_values[0][int(1e6):int(1.01e6)], label='Pol2 0')
    ax.plot(x, all_values[1][int(1e6):int(1.01e6)], label='CPD 0')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    fig.canvas.flush_events()

    for time in range(10000):
        pol_new = reaction_pol(pol, cpd, dt=dt)
        cpd_new = reaction_cpd(cpd, pol, dt=dt)
        cpd = cpd_new
        pol = pol_new

        line_pol.set_ydata(pol)
        line_cpd.set_ydata(cpd)
        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == '__main__':
    main()

