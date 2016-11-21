import numpy as np
import librosa
from scipy.io.wavfile import *
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

## file paths
filename_wind="SR006.wav"
filename_mixture="Wind_Bob_3.wav"

## audio parameters
sample_rate=16000
n_fft=1024
hop_size=512

##training parameters
gamma=2

## load files
[sr_w, wind_signal]=read(filename_wind)
[sr_m, mixture]=read(filename_mixture)

## fft preprocessing
wind_stft=librosa.core.stft(wind_signal,n_fft,hop_size)
mixture_stft=librosa.core.stft(mixture,n_fft,hop_size)

log_spec_wind=librosa.logamplitude(np.abs(wind_stft))**gamma
log_spec_mixture=librosa.logamplitude(np.abs(mixture_stft))**gamma

## plot spectrograms
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
librosa.display.specshow(log_spec_wind,sr_w,hop_size,x_axis="time", y_axis="log")
plt.subplot(1,2,2)
librosa.display.specshow(log_spec_mixture,sr_m,hop_size,x_axis="time", y_axis="log")
plt.show()

## nmf






