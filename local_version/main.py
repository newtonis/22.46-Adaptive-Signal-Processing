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

    rows = 2
    cols = 3

    fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
    fig.suptitle('Curvas para comparar periodograma')

    N_fft = 2**4
    test = strain[0:N_fft]
    test2 = strain[N_fft:N_fft*2]

    M = getPeriodogram(test)
    M2 = getPeriodogram(test2)

    x = test
    f, Pxx_den = signal.periodogram(x, fs)

    freq_axis = np.arange(0, fs, fs / N_fft)

    # diccionario de curvas con titulo, x, y,  si es en funcion de freq y posicion
    curvas = {'D1. Long': (range(len(test)), test,False,(0, 0)),
              'Sxx1^  Long:' + str(len(M)): (freq_axis, M, True,(1, 0)),
              'Sxx1^ con signal. Long:' + str(len(Pxx_den)):(f, Pxx_den, True,(1, 1)),
              'D2. Long:' + str(len(test2)):(range(len(test2)),test2,False,(0, 2)),
              'Sxx2^. Long:' + str(len(M2)):(freq_axis, M2, True,(1, 2))
              }

    # ploteamos las curvas segun corresponda
    for title, meta in curvas.items():
        x, y, isXinFreq, pos = meta
        axs[pos].title.set_text(title)
        if isXinFreq:
            axs[pos].semilogy(x, y)
        else:
            axs[pos].plot(x, y)

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.show()


