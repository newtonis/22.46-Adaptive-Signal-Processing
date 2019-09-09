import numpy as np
import readligo as rl
import matplotlib.pyplot as plt
from scipy import signal


def getWindowedArray(time, strain, N):
    time_w = [time[i:i + N] for i in range(windows_qnt)]
    strain_w = [strain[i:i + N] for i in range(windows_qnt)]
    # falta la ultima parte
    time_w.append(time[(windows_qnt - 1) + N:len(time)])  # agrego lo que faltaba
    strain_w.append(strain[(windows_qnt - 1) + N:len(strain)])  # agrego lo que faltaba
    return time_w, strain_w

def getPeriodogram(data_array):
    aux = np.fft.fft(data_array)
    sxx_hat = [abs(x) ** 2 / len(aux) for x in aux]
    return sxx_hat


if __name__ == '__main__':
    Hanford_32 = "data\\H1_32.hdf5"

    fs = 4096
    strain, time, dq = rl.loaddata(Hanford_32, 'H1') # L1  para el otro caso

    total_time = time[-1]-time[0]

    dt = 0.02
    windows_qnt = int(total_time/dt)

    # decidimos la longitud de la ventana
    N = len(time) // windows_qnt

    time_w, strain_w = getWindowedArray(time, strain, N)

    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    fig.suptitle('Horizontally stacked subplots')

    N_fft = 2**4
    test = strain[0:N_fft]
    test2 = strain[N_fft:N_fft*2]

    M = getPeriodogram(test)
    M2 = getPeriodogram(test2)

    x = test
    f, Pxx_den = signal.periodogram(x, fs)

    freq_axis = np.arange(0, fs, fs / N_fft)

    axs[0, 0].plot(range(len(test)), test)
    axs[0, 0].title.set_text('D1. Long:'+str(len(test)))

    axs[1, 0].semilogy(freq_axis, M)
    axs[1, 0].title.set_text('Sxx1^  Long:' + str(len(M)))

    axs[1, 1].semilogy(f, Pxx_den)
    axs[1, 1].title.set_text('Sxx1^ con signal. Long:' + str(len(Pxx_den)))

    axs[0, 2].plot(range(len(test2)), test2)
    axs[0, 2].title.set_text('D2. Long:' + str(len(test2)))

    axs[1, 2].semilogy(freq_axis, M2)
    axs[1, 2].title.set_text('Sxx2^. Long:'+str(len(M2)))


    left = 0.125  # the left side of the subplots of the figure
    right = 0.9   # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
                  # expressed as a fraction of the average axis width
    hspace = 0.2  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.show()


