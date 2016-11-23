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

# ######### PRE-PROCESSING: ########################

# # Inputs:
# filename_wind = "data/test_raw/wind/wind2.wav"
# filename_speech = "data/test_raw/female1.wav"
# SNRin = 0 # input SNR

# # pre-processing:
# (y,x,wn) = pre_processing(filename_speech, filename_wind, SNRin)

fs = 16000

#############  STFT  #########################

# # audio parameters
sample_rate=16000
n_fft=512 # 32 ms frame (like in paper)
hop_size=128 # 75% overlap


# ##training parameters
gamma=2

# ## fft preprocessing
# wn_stft=librosa.core.stft(wn,n_fft,hop_size)
# y_stft=librosa.core.stft(y,n_fft,hop_size)

# log_spec_wn=librosa.logamplitude(np.abs(wn_stft))**gamma
# log_spec_y=librosa.logamplitude(np.abs(y_stft))**gamma

## plot spectrograms
# plt.figure(figsize=(12, 8))
# plt.subplot(1,2,1)
# librosa.display.specshow(log_spec_wn,fs,hop_size,x_axis="time", y_axis="log")
# plt.subplot(1,2,2)
# librosa.display.specshow(log_spec_y,fs,hop_size,x_axis="time", y_axis="log")
# plt.show()

############# TRAIN DICTIONARY #################


filename_train_wind = "data/train/wind_train_1min.wav"

[fs, wn_train]=read(filename_train_wind)

t_train = 10 # training time in second

L = fs * t_train

wn_train = wn_train[:L]

Y_train=librosa.core.stft(wn_train,n_fft,hop_size)

Y_mag, Y_phase = librosa.magphase(Y_train)

S_train = Y_mag**gamma

(n_frequencies, n_frames) = S_train.shape

# plt.figure(figsize=(12, 8))
# log_spec_wn_train=librosa.logamplitude(np.abs(wn_train_stft))**gamma
# librosa.display.specshow(log_spec_wn_train,fs,hop_size,x_axis="time", y_axis="log")

# Parameters:
n_atoms = 64
sparsity_degree = 16

# Initialize dictionary:
D = abs(np.random.rand(n_frequencies, n_atoms))
X = np.zeros((n_atoms, n_frames))

# Load previously conputed dictionary:
# D = pickle.load( open( "Dictionary_4atoms_10it.npy", "rb" ) )

# iterate:
Nit = 10

for it in range(Nit):
	print("iteration:", it)

	# sparse coding:

	for t in range(n_frames):

		# print(t)

		current_frame = S_train[:,t]

		omp = OrthogonalMatchingPursuit(n_atoms)
		omp.fit(D, current_frame)
		X[:, t] = omp.coef_
		# print(X.shape)



	# dictionary learning:

	# pseudo_inv = np.dot(X.transpose(), np.linalg.inv(np.dot(X, X.transpose())))
	# D = np.dot(S_train, pseudo_inv) 

	D = np.linalg.lstsq(X.transpose(), S_train.transpose(), 1e-8)

	D = D[0] # solution is first element of tuple

	# # Normalize D:
	# a = np.sum(np.abs(D**2),axis=-1)**(1./2)	
	# D = np.dot(np.diag(1/a),D)

	D = D.transpose()


	# print(D.shape)
	# print(X.shape)



# PLot dictionary:

plt.figure(figsize=(12, 8))
librosa.display.specshow(np.log(D+eps),fs,hop_size,x_axis="time", y_axis="log")

# Reconstructed spectrogram:

S_reconst = np.dot(D,X)
# Y_reconst = np.sqrt(S_reconst)*np.exp(1j * Y_phase)
# y_reconst = librosa.core.istft(Y_reconst,n_fft,hop_size)

# scipy.io.wavfile.write("y_reconst.wav", fs, y_reconst)

plt.figure(figsize=(12, 8))
librosa.display.specshow(np.log(S_train+eps),fs,hop_size,x_axis="time", y_axis="log")

plt.figure(figsize=(12, 8))
librosa.display.specshow(np.log(S_reconst+eps),fs,hop_size,x_axis="time", y_axis="log")
plt.show()




# Save/load dict:

# save:
pickle.dump(D, open( 'Dictionary_16atoms_10it_10min.npy', 'wb'))
pickle.dump(X, open( 'Activations_16atoms_10it_10min.npy', 'wb'))


# load:
# D_loaded = pickle.load( open( "Dictionary_4atoms_10it.npy", "rb" ) )




################ TEST  #############################

