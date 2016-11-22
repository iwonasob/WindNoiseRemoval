import numpy as np
import librosa
from scipy.io.wavfile import *
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns


def pre_processing(filename_speech, filename_wind, SNRin):

	######### PRE-PROCESSING: ########################

	# usage:
	# (y,x,wn) = pre_processing(filename_speech, filename_wind, SNRin)

	# load files
	[sr_wn, wn]=read(filename_wind)
	[sr_x, x]=read(filename_speech)

	# Normalize
	x = x/np.max(abs(x));
	wn = wn/np.max(abs(wn));

	# Crop signals so that they have same size
	Lx = x.size
	Lwn = wn.size
	wn = wn[0:Lx] # crop signal

	# Normalize wind to desired SNR:
	wn = wn * np.sqrt(np.sum(x**2)/(np.sum(wn**2)*(10**(SNRin/10))))

	# Create mixture:
	y = x + wn # Noisy signal


	return(y,x,wn)