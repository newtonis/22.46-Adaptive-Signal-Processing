import numpy as np
import readligo as rl
import matplotlib.pyplot as plt
import math

def getPeriodogram(data_array, fs, N_fft):
    aux = np.fft.fft(data_array)
    sxx_hat = [abs(x) ** 2 / len(aux) for x in aux]
    new_size = math.ceil((len(data_array)+1) / 2)
    init_freq = 10
    step = (fs/2) / new_size
    init_index = int(init_freq/step)
    if not np.iscomplex(data_array).all():
        sxx_hat = sxx_hat[0:new_size]
        freq_axis = np.arange(0, fs/2, (fs/2) / new_size) # tomo la parte positiva
        freq_axis = freq_axis[init_index:]
        sxx_hat = sxx_hat[init_index:]
    else:
        freq_axis = np.arange(-fs/2 + fs/N_fft, fs/2, fs / N_fft)
    return freq_axis, sxx_hat


if __name__ == '__main__':
    fn_H1 = "data\\H1_32.hdf5"
    # fs = 4096
    #     # strain, time, dq = rl.loaddata(fn_H1, 'H1') # L1  para el otro caso
    #     # #N_fft = len(time)
    #     # N_fft = fs
    #     # f, M = getPeriodogram(strain, fs, N_fft)
    #     # aux = [(x/10000)**(1/2) for x in M]
    #     # # # usaron mas datos y tambien usaron whitening
    #     # plt.loglog(f, aux, 'b', label='H1_periodogram')
    #     # strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
    #     # NFFT = 1 * fs
    #     # fmin = 10
    #     # fmax = 2000
    #---------------------
    # Read in strain data
    #---------------------
    strain, time, channel_dict = rl.loaddata(fn_H1, 'H1')
    ts = time[1] - time[0] #-- Time between samples
    fs = int(1.0 / ts)          #-- Sampling frequency

    #---------------------------------------------------------
    # Find a good segment, get first 16 seconds of data
    #---------------------------------------------------------
    segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'], fs)
    length = 16  # seconds
    strain_seg = strain[segList[0]][0:(length*fs)]
    time_seg = time[segList[0]][0:(length*fs)]

    f, M = getPeriodogram(strain_seg, fs, len(strain_seg))
    # # usaron mas datos y tambien usaron whitening
    plt.loglog(f, np.sqrt(M), 'b', label='H1_periodogram')

    #------------------------------------------
    # Apply a Blackman Window, and plot the FFT
    #------------------------------------------
    window = np.blackman(strain_seg.size)
    windowed_strain = strain_seg*window
    freq_domain = np.fft.rfft(windowed_strain) / fs
    freq = np.fft.rfftfreq(len(windowed_strain))*fs

    plt.loglog( freq, abs(freq_domain), 'r')
    plt.axis([10, fs/2.0, 1e-55, 1e-18])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Strain / Hz')

    #----------------------------------
    # Make PSD for first chunk of data
    #----------------------------------
    # Pxx, freqs = mlab.psd(strain_seg, Fs = fs, NFFT=fs)
    # plt.loglog(freqs, Pxx)
    # plt.axis([10, 2000, 1e-46, 1e-36])
    # plt.grid('on')
    # plt.ylabel('PSD')
    # plt.xlabel('Freq (Hz)')

    plt.show()
