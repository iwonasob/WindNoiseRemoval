import numpy as np
#import librosa
import scipy.io.wavfile

#from scipy import signal
#import matplotlib.pyplot as plt
#import seaborn as sns


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