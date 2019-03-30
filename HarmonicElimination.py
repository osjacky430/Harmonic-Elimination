import matplotlib.pyplot as plt
import numpy as np


pulses = [1, 2, 4, 20]
harmonics = [5, 7, 11, 13, 17, 19, 23, 25, 29, 31]

rpw_points = np.linspace(0.001, 1, 100)


def calculate_fundamental_magnitude(rpw: list, pulse: int, pulse_duration: float):
    ret_val = []

    for rpw_ in rpw:

        a1 = 0
        b1 = 0
        for k in range(0, 2 * pulse):
            start = pulse / 2 + k

            a1 += 2 * (np.cos(pulse_duration * start) - np.cos(pulse_duration * (start + rpw_))) / np.pi
            b1 += 2 * (np.sin(pulse_duration * (start + rpw_)) - np.sin(pulse_duration * start)) / np.pi

        ret_val.append(np.sqrt(a1 ** 2 + b1 ** 2))

    return ret_val


def calculate_magnitude(harmonic: int, rpw: float, pulse: int, pulse_duration: float):
    an = 0
    bn = 0

    coeff = harmonic * pulse_duration
    for k in range(0, 2 * pulse):
        start = pulse / 2 + k

        an += 2 * (np.cos(coeff * start) - np.cos(coeff * (start + rpw))) / (harmonic * np.pi)
        bn += 2 * (np.sin(coeff * (start + rpw)) - np.sin(coeff * start)) / (harmonic * np.pi)

    return np.sqrt(an**2 + bn**2)


if __name__ == '__main__':

    for i, m in enumerate(pulses):

        # print('---------- Pulse wave %d ------------' % (m * 2))

        Delta = np.pi / (3 * m)

        c1 = calculate_fundamental_magnitude(rpw=rpw_points, pulse=m, pulse_duration=Delta)

        for n in harmonics:

            ratio = []

            # plot figure setup
            plt.figure(i)
            plt.xticks(np.linspace(0, 1, num=11, endpoint=True))
            plt.xlim(0, 1)
            plt.xlabel('Relative Pulse Width  ' r'$\frac{\delta}{\Delta}$')
            plt.ylim(0, 1.2)
            plt.ylabel('Fundamental Magnitude / ' r'$V_{dc}$')
            plt.title('%d pulses per half-cycle' % (2 * m))

            for j, relative_pulse_width in enumerate(rpw_points):
                cn = calculate_magnitude(harmonic=n, rpw=relative_pulse_width, pulse=m, pulse_duration=Delta)

                ratio_ = (cn + 1e-6) / (c1[j] + 1e-6)
                ratio.append(ratio_)

            plt.plot(rpw_points, ratio, label='%d-th Harmonic' % n)

        plt.plot(rpw_points, c1, label='Fundamental')
        plt.grid()
        plt.legend()

    plt.show()
