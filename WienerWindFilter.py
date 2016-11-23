# -*- coding: utf-8 -*-

import numpy as np

import os.path
from pre_processing import pre_processing

def load_train_sample(sample_id=0, snr=-10):
    wind_f_stem = os.path.join('data', 'train', 'wind_train_')
    voice_f_stem = os.path.join('data', 'test_raw', 'obama')
    
    if sample_id == 0:
        wind_f = wind_f_stem + '1min.wav'
        voice_f = voice_f_stem + '1min.wav'
    elif sample_id == 1:
        wind_f = wind_f_stem + '1min.wav'
        voice_f = voice_f_stem + '1min.wav'
    elif sample_id == 2:
        wind_f = wind_f_stem + '1min.wav'
        voice_f = voice_f_stem + '1min.wav'
    
    x, x_noisy, x_fs = pre_processing(voice_f, wind_f, snr)

    return x, x_noisy, x_fs

class WienerWindFilter:
    def __init__(self, train_sample_id=0, train_snr=-10):
        x, x_noisy, x_fs = load_train_sample(train_sample_id, train_snr)
        
        n = x_fs
        X = np.fft.rfft(x, n)
        XN = np.fft.rfft(x_noisy, n)

        # Estimate the filter weights.
        self.H = abs(X)**2 / abs(XN)**2

        # Sometimes this proceedure is unstable, prevent it.
        self.H = np.minimum(self.H, 1)

#    def apply(self, x):
#        X = np.fft.rfft(x)
#
#        Y = X * self.H
#
#        y = np.fft.irfft(Y)#, n)
#
#        return y

    def apply(self, x, step=128):
        n = (len(self.H) - 1) * 2
        xn = len(x)

        y = np.zeros(xn)
        w = np.zeros(xn)
        for xi in range(0, xn - n + 1, step):
            X = np.fft.rfft(x[xi:(xi+n)])
            Y = X * self.H
            #y[xi:(xi+n)] = y[xi:(xi+n)] + np.fft.irfft(Y)
            y[xi:(xi+n)] += np.fft.irfft(Y)
            w[xi:(xi+n)] += 1

        # Weight.
        y = y / w

        return y