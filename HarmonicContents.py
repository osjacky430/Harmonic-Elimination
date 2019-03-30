from functools import partial
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np


def is_descending_order(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i+1]:
            return False
    return True


def check_convergence(epsilon, F_old):
    return np.all(np.abs(F_old) < epsilon)


def update_iteration_matrix(to_approach: dict, alpha: np.array):
    M = len(alpha)
    dFdA = np.ndarray(shape=(M, M))
    F = np.zeros(shape=(M, 1))

    for i, n in enumerate(to_approach):
        for k in range(M):
            dFdA[i, k] = 2 * n * np.sin(n * alpha[k]) * (-1)**k
            F[i] = F[i] + 2 * (-1)**(k + 1) * np.cos(n * alpha[k]) - n * np.pi * to_approach[n] / (8 * M)

    F = F + 1
    return dFdA, F


def newton_ralphson(to_approach: dict, initial_guess: np.array, convergence_satisfied: callable):

    A = initial_guess
    update = partial(update_iteration_matrix, to_approach)
    dFdA, F_old = update(A)
    F_new = np.ndarray(shape=F_old.shape)
    iter_count = 0

    while not convergence_satisfied(F_old=F_old) and iter_count < 20000:
        F_new = F_old
        dA = - np.dot(np.linalg.inv(dFdA), F_new)
        A = A + dA

        dFdA, F_old = update(A)

        iter_count = iter_count + 1

    if not is_descending_order(A):
        print('Iteration fail: unreasonable sequence')

    print('Iteration over: ', iter_count)
    print('Iteration converged to values: ', F_new.reshape(-1).tolist())
    print('Resulting switch timing:', np.rad2deg(A.reshape(-1).tolist()))

    return A


def generate_pulses(phase_shift: list):
    points = 10000
    wt = np.linspace(0, 1/4, num=points)
    output_half = []

    for i, val in enumerate(wt):
        if (len(phase_shift) >= 2 and phase_shift[0] / (2 * np.pi) < val < phase_shift[1] / (2 * np.pi)) or \
                (len(phase_shift) == 1 and phase_shift[0] / (2 * np.pi) < val):
            output_half.append(-1)
        else:
            if len(output_half) != 0 and output_half[-1] == -1:
                phase_shift.pop(0)
                phase_shift.pop(0)
            output_half.append(1)

    output = output_half + list(reversed(output_half))
    output = output + list(reversed(np.multiply(-1, output)))
    return output, np.linspace(0, 1, num=points * 4)


if __name__ == '__main__':
    converge_func = partial(check_convergence, 1e-10)

    final = newton_ralphson(
                to_approach={5: 0, 7: 0},
                initial_guess=np.deg2rad([6.7, 17.2]).reshape(2, 1),
                convergence_satisfied=converge_func)

    result, time_line = generate_pulses(final.reshape(-1).tolist())

    spectrum = np.abs(fftpack.fft(result)) / len(result)
    fundamental_mag = max(spectrum)
    spectrum = spectrum / fundamental_mag
    freq = fftpack.fftfreq(len(result), 1 / len(result))

    print('Fundamental Magnitude', fundamental_mag)

    plt.figure(1)

    plt.plot(time_line, result)
    plt.xlim(0, 1)
    plt.title('No 5-th, and 7-th Harmonic Waveform')
    plt.xlabel('phase angle')
    plt.grid()

    fig, ax = plt.subplots()

    plt.xlim(0, 32)
    plt.xticks(np.arange(1, 33, step=2))
    plt.yticks(np.linspace(0, 1, num=10))
    plt.ylim(0, 1.05)

    ax.stem(freq[0:32], spectrum[0: 32])
    plt.show()
