import numpy as np
import librosa
from scipy.io.wavfile import *
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

from pre_processing import *

import pickle

eps = np.finfo(float).eps

######### PRE-PROCESSING: ########################

# Inputs:
filename_wind = "data/test_raw/wind/wind2.wav"
filename_speech = "data/test_raw/female1.wav"
SNRin = 0 # input SNR

# pre-processing:
(y,x,wn) = pre_processing(filename_speech, filename_wind, SNRin)

fs = 16000

#############  STFT  #########################

# # audio parameters
sample_rate=16000
n_fft=512 # 32 ms frame (like in paper)
hop_size=128 # 75% overlap


# ##training parameters
gamma=2

## fft preprocessing
wn_stft=librosa.core.stft(wn,n_fft,hop_size)
y_stft=librosa.core.stft(y,n_fft,hop_size)

log_spec_wn=librosa.logamplitude(np.abs(wn_stft))**gamma
log_spec_y=librosa.logamplitude(np.abs(y_stft))**gamma

## plot spectrograms
# plt.figure(figsize=(12, 8))
# plt.subplot(1,2,1)
# librosa.display.specshow(log_spec_wn,fs,hop_size,x_axis="time", y_axis="log")
# plt.subplot(1,2,2)
# librosa.display.specshow(log_spec_y,fs,hop_size,x_axis="time", y_axis="log")
# plt.show()


## LOAD DICTIONARY

# Load previously conputed dictionary:
D = pickle.load( open( "Dictionary_4atoms_10it.npy", "rb" ) )


## Decompose

Y_mixture=librosa.core.stft(y,n_fft,hop_size)

Y_mixture_mag, Y_mixture_phase = librosa.magphase(Y_mixture)

S_mixture = Y_mixture_mag**gamma

(n_frequencies, n_frames) = S_mixture.shape

# plt.figure(figsize=(12, 8))
# log_spec_wn_train=librosa.logamplitude(np.abs(wn_train_stft))**gamma
# librosa.display.specshow(log_spec_wn_train,fs,hop_size,x_axis="time", y_axis="log")

# Parameters:
n_atoms_tot = 128
n_atom = int(n_atoms_tot/2)
sparsity_degree_tot = 32

# Initialize dictionary:
D_s = abs(np.random.rand(n_frequencies, n_atom))
X = np.zeros((n_atoms_tot, n_frames))

# Load previously conputed dictionary:
D_w = pickle.load( open( "Dictionary_16atoms_10it_10min.npy", "rb" ) )

# D_w = D_w.transpose()
# a = np.sum(np.abs(D_w**2),axis=-1)**(1./2)	
# D_w = np.dot(np.diag(1/a),D_w)
# D_w = D_w.transpose()


# iterate:
Nit = 4

for it in range(Nit):
	print("iteration:", it)

	# sparse coding:

	for t in range(n_frames):

		# print(t)

		current_frame = S_mixture[:,t]

		D = np.concatenate((D_s, D_w), axis=1)


		omp = OrthogonalMatchingPursuit(n_atoms_tot)
		omp.fit(D, current_frame)
		X[:, t] = omp.coef_
		# print(X.shape)



	# dictionary learning:

	X_s = X[:n_atom, :]

	# pseudo_inv = np.dot(X_s.transpose(), np.linalg.inv(np.dot(X_s, X_s.transpose())))
	S_wind_curr = np.dot(D[:, :n_atom], X[n_atom:, :])
	# D = np.dot(S_mixture - S_wind_curr, pseudo_inv)
	# D_s = np.dot(S_mixture - S_wind_curr, pseudo_inv)


	D_s_tuple = np.linalg.lstsq(X_s.transpose(), (S_mixture - S_wind_curr).transpose(), 1e-8)

	D_s = D_s_tuple[0] # solution is first element of tuple

	# # Normalize D:
	# a = np.sum(np.abs(D_s**2),axis=-1)**(1./2)	
	# D_s = np.dot(np.diag(1/a),D_s)
	D_s = D_s.transpose()


D = np.concatenate((D_s, D_w), axis=1)

print(D.shape)
print(X.shape)

# PLot dictionary:

plt.figure(figsize=(12, 8))
librosa.display.specshow(np.log(D+eps),fs,hop_size,x_axis="time", y_axis="log")

# Reconstructed spectrogram:

S_reconst = np.dot(D,X)
# Y_reconst = np.sqrt(S_reconst)*np.exp(1j * Y_phase)
# y_reconst = librosa.core.istft(Y_reconst,n_fft,hop_size)

# scipy.io.wavfile.write("y_reconst.wav", fs, y_reconst)

plt.figure(figsize=(12, 8))
librosa.display.specshow(np.log(S_mixture+eps),fs,hop_size,x_axis="time", y_axis="log")

plt.figure(figsize=(12, 8))
librosa.display.specshow(np.log(S_reconst+eps),fs,hop_size,x_axis="time", y_axis="log")
plt.show()



