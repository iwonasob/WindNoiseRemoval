import numpy as np
import os.path
import librosa
import scipy.io.wavfile
from scipy.io.wavfile import *



def pre_processing(filename_speech, filename_wind, SNRin):

	######### PRE-PROCESSING: ########################

	# usage:
	# filename_wind = "data/test_raw/wind/wind2.wav"
	# filename_speech = "data/test_raw/female1.wav"
	# SNRin = 0 # input SNR
	# (y,x,wn) = pre_processing(filename_speech, filename_wind, SNRin)

	# load files
	[sr_wn, wn]=scipy.io.wavfile.read(filename_wind)
	[sr_x, x]=scipy.io.wavfile.read(filename_speech)

	# Normalize
	x = x/np.max(abs(x));
	wn = wn/np.max(abs(wn));

	# Crop signals so that they have same size
	Lx = x.size
	Lwn = wn.size
	L = min(Lx,Lwn)

	wn = wn[0:L] # crop signal
	x = x[0:L]

	# Normalize wind to desired SNR:
	wn = wn * np.sqrt(np.sum(x**2)/(np.sum(wn**2)*(10**(SNRin/10))))

	# Create mixture:
	y = x + wn # Noisy signal

	return(x, y, sr_x)

	# # Outputs:
	# y: mixture
	# x: clean speech
	# wn: wind

def prepare_training_data():
	filename_wind_train = os.path.join('data', 'train', 'wind_train_1min.wav')

	## audio parameters
	n_fft = 512
	hop_size = 128
	gamma = 2

	[sr, w_train ] = read(filename_wind_train)
	wind_stft = librosa.core.stft(w_train, n_fft, hop_size)
	magnitude_wind=np.abs(wind_stft)**gamma
	magnitude_wind_norm=magnitude_wind/np.max(magnitude_wind)
	return magnitude_wind_norm, sr, n_fft, hop_size, gamma