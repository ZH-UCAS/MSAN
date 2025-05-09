
import numpy as np
# import re
# import stft
import seaborn as sns
from scipy.signal import butter, lfilter
from scipy import signal
import mne
import matplotlib.pyplot as plt
import sys

sys.path.append("..")


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    # print("data", data, "lowcut", lowcut, "highcut", highcut, "fs", fs, "order", order)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # print("low", low, "high", high)
    i, u = butter(order, [low, high], btype='bandstop')
    # print("i", i, "u", u)
    y = lfilter(i, u, data)
    # print("y", y)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y


def getSpectral_STFT_T(data, fs=256):
    # (7680, 23)
    #	print("stft input {}".format(data.shape))
    # 	fs=256
    lowcut = 117
    highcut = 123
    overlap = 0.75 * fs

    y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    # print(fs)
    # print(int(overlap))
    # y 18 7680

    Pxx = signal.spectrogram(y, nfft=fs, fs=fs, return_onesided=True, noverlap=192)[2]  # (23, 129, 34)
    Pxx = np.transpose(Pxx, (0, 2, 1))  # (23, 34, 129)
    Pxx = np.concatenate((Pxx[:, :, 1:57],
                          Pxx[:, :, 64:117],
                          Pxx[:, :, 124:]), axis=-1)  # (23, 34, 114)

    # 归一化(33, 129, 23)
    stft_data = (10 * np.log10(Pxx) - (10 * np.log10(Pxx)).min()) / (10 * np.log10(Pxx)).ptp()
    # print("stft result {}".format(stft_data.shape))#(1, 95, 114)
    # print(stft_data.shape)
    return stft_data  # (23, 59, 114)


def getSpectral_STFT(data, fs=256):
    # (7680, 23)
    #	print("stft input {}".format(data.shape))
    # 	fs=256
    # print(data)
    lowcut = 117
    highcut = 123
    overlap = 0.75 * fs

    y = butter_bandstop_filter(data.T, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    # print("y1")

    # print(y)
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)
    # print("y2")
    # print(y)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)
    # print("y3")
    # print(y)

    # print(fs)
    # print(int(overlap))
    # y 18 7680
    # print("Pxx")

    Pxx = signal.spectrogram(y, nfft=fs, fs=fs, return_onesided=True, noverlap=192)[2]  # (23, 129, 34)
    # print(Pxx)

    Pxx = np.transpose(Pxx, (0, 2, 1))  # (23, 34, 129)
    Pxx = np.concatenate((Pxx[:, :, 1:57],
                          Pxx[:, :, 64:117],
                          Pxx[:, :, 124:]), axis=-1)  # (23, 34, 114)
    # print("Pxx2")

    # print(Pxx)

    # 归一化(33, 129, 23)
    stft_data = (10 * np.log10(Pxx) - (10 * np.log10(Pxx)).min()) / (10 * np.log10(Pxx)).ptp()
    # print("stft result {}".format(stft_data.shape))#(1, 95, 114)
    # print("stft_data")
    # print(stft_data)
    # print(data)
    # print(stft_data.shape)

    return stft_data  # (23, 59, 114)



def get_XW_Spectral_STFT_T(data, fs=256):
    # (7680, 23)
    #	print("stft input {}".format(data.shape))
    # 	fs=256
    lowcut = 117
    highcut = 123
    overlap = 0.75 * fs

    y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    # print(fs)
    # print(int(overlap))
    # y 18 7680

    Pxx = signal.spectrogram(y, nfft=fs, fs=fs, return_onesided=True, noverlap=192)[2]  # (23, 129, 34)
    Pxx = np.transpose(Pxx, (0, 2, 1))  # (23, 34, 129)
    Pxx = np.concatenate((Pxx[:, :, 1:57],
                          Pxx[:, :, 64:117],
                          Pxx[:, :, 124:]), axis=-1)  # (23, 34, 114)

    # 归一化(33, 129, 23)
    stft_data = (10 * np.log10(Pxx) - (10 * np.log10(Pxx)).min()) / (10 * np.log10(Pxx)).ptp()
    # print("stft result {}".format(stft_data.shape))#(1, 95, 114)
    # print(stft_data.shape)
    stft_data = stft_data.astype(np.float32)
    return stft_data  # (23, 59, 114)


def get_XW_Spectral_STFT(data, fs=256):
    # (7680, 23)
    #	print("stft input {}".format(data.shape))
    # 	fs=256
    # print(data)
    lowcut = 117
    highcut = 123
    overlap = 0.75 * fs

    y = butter_bandstop_filter(data.T, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    # print("y1")

    # print(y)
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)
    # print("y2")
    # print(y)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)
    # print("y3")
    # print(y)

    # print(fs)
    # print(int(overlap))
    # y 18 7680
    # print("Pxx")

    Pxx = signal.spectrogram(y, nfft=fs, fs=fs, return_onesided=True, noverlap=192)[2]  # (23, 129, 34)
    # print(Pxx)

    Pxx = np.transpose(Pxx, (0, 2, 1))  # (23, 34, 129)
    Pxx = np.concatenate((Pxx[:, :, 1:57],
                          Pxx[:, :, 64:117],
                          Pxx[:, :, 124:]), axis=-1)  # (23, 34, 114)
    # print("Pxx2")

    # print(Pxx)

    # 归一化(33, 129, 23)
    stft_data = (10 * np.log10(Pxx) - (10 * np.log10(Pxx)).min()) / (10 * np.log10(Pxx)).ptp()
    # print("stft result {}".format(stft_data.shape))#(1, 95, 114)
    # print("stft_data")
    # print(stft_data)
    # print(data)
    # print(stft_data.shape)
    stft_data = stft_data.astype(np.float32)
    return stft_data  # (23, 59, 114)



def getSpectral_Morlet(data, fs):
    if len(data.shape) > 1:
        data = data.squeeze()
    # t, dt = np.linspace(0, 1, 200, retstep=True)#t.shape(200,) dt=0.005025125628140704
    # fs = 1/dt #199
    w = 6.
    # sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)#(200,)
    freq = np.linspace(1, fs / 2, 100)  # (100,)
    widths = w * fs / (2 * freq * np.pi)  # (100,)
    cwtm = signal.cwt(data, signal.morlet2, widths, w=w)  # (100, 200) (freq, time)
    # plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
    # plt.show()
    return cwtm
