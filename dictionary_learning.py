import numpy as np
import librosa
from scipy.io.wavfile import *
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns


######### PRE-PROCESSING: ########################

filename_wind = "../data/test/wind/wind1.wav"
filename_speech = "../data/test/female1.wav"

## load files
[sr_wn, wn]=read(filename_wind)
[sr_x, x]=read(filename_speech)

SNRin = 5

# normalize:
wn = wn * np.sqrt(np.sum(x**2)/(np.sum(wn**2)*10**(SNRin/10)))


y = x + wn # Noisy signal


#############  STFT  #########################


# # audio parameters
# sample_rate=16000
# n_fft=256 # 16 ms frame
# hop_size=128


# ##training parameters
# gamma=2



# ## fft preprocessing
# wind_stft=librosa.core.stft(wind_signal,n_fft,hop_size)
# mixture_stft=librosa.core.stft(mixture,n_fft,hop_size)

# log_spec_wind=librosa.logamplitude(np.abs(wind_stft))**gamma
# log_spec_mixture=librosa.logamplitude(np.abs(mixture_stft))**gamma

# ## plot spectrograms
# plt.figure(figsize=(12, 8))
# plt.subplot(1,2,1)
# librosa.display.specshow(log_spec_wind,sr_w,hop_size,x_axis="time", y_axis="log")
# plt.subplot(1,2,2)
# librosa.display.specshow(log_spec_mixture,sr_m,hop_size,x_axis="time", y_axis="log")
# plt.show()

## nmf



## Output

SNRout = 10*np.log(np.sum(x**2)/(np.sum(wn**2)))
