#!/usr/bin/python3
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from ODE import ODESymmetric, nabla_sq_1d
from fitting import fit

VERBOSITY = 3


def main():
    # Original parameters A
    # 1, -1, -0.1, 1
    # Original parameters B
    # 1, -1, -0.1, 3
    A = np.genfromtxt('data/A-rand.csv', delimiter=',')
    B = np.genfromtxt('data/B-rand.csv', delimiter=',')
    x = np.genfromtxt('data/Time.csv', delimiter=',')

    ode_A = ODESymmetric(
        np.random.random(4),
        np.asarray([False, False, False, False]),
        num_sys=A.shape[1],
        spatial_d=nabla_sq_1d,
        auto_pow=3
    )
    ode_B = ODESymmetric(
        np.random.random(4),
        np.asarray([False, False, False, False]),
        num_sys=B.shape[1],
        spatial_d=nabla_sq_1d,
        auto_pow=3
    )
    ODESymmetric.own_idx = 0

    ode_A, ode_B = fit(
        np.asarray([A[:-1], B[:-1]]),  # Layout (species, time points, spatial dimension)
        [ode_A, ode_B],
        x=x,
        w_model=1.,
        degree=4,  # Finding ===> Is very sensitive to the degree. Using degree of 4 or 5 works perfect
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

    a_param = np.mean(ode_A.coeff, axis=0)
    b_param = np.mean(ode_B.coeff, axis=0)
    ode_A.set_uniform_coeff(a_param)
    ode_B.set_uniform_coeff(b_param)

    if VERBOSITY > 0:
        print('A params %s, B params %s' % (a_param, b_param))

    a = A[0, :].reshape(1, A.shape[1])
    b = B[0, :].reshape(1, B.shape[1])

    a_history = [a[0]]
    b_history = [b[0]]
    for _ in np.arange(3000):
        a_dev = ode_A.calc(np.asarray([a, b]), dt=0.1)
        b_dev = ode_B.calc(np.asarray([a, b]), dt=0.1)
        a += a_dev
        b += b_dev
        a_history.append(np.copy(a[0]))
        b_history.append(np.copy(b[0]))

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0][0].pcolor(np.asarray(A), vmin=A.min(), vmax=A.max())
    ax[0][0].set_title('Species A')
    ax[0][0].set_xlabel('Position in x')
    ax[0][0].set_ylabel('Time step')
    ax[0][1].pcolor(np.asarray(B), vmin=B.min(), vmax=B.max())
    ax[0][1].set_title('Species B')
    ax[0][1].set_xlabel('Position in x')
    ax[0][1].set_ylabel('Time step')
    amin, amax = np.asarray(a_history).min(), np.asarray(a_history).max()
    bmin, bmax = np.asarray(b_history).min(), np.asarray(b_history).max()
    ax[1][0].set_title('Appr. Species A')
    ax[1][0].set_xlabel('Position in x')
    ax[1][0].set_ylabel('Time step')
    ax[1][0].pcolor(np.asarray(a_history), vmin=amin, vmax=amax)
    ax[1][1].set_title('Appr. Species B')
    ax[1][1].set_xlabel('Position in x')
    ax[1][1].set_ylabel('Time step')
    ax[1][1].pcolor(np.asarray(b_history), vmin=bmin, vmax=bmax)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

