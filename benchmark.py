# -*- coding: utf-8 -*-

import numpy as np
import os.path
from pre_processing import pre_processing
from compute_snr import compute_snr
import scipy.io.wavfile

# Example of how to define f
# Simple case, no extra parameters:
#   f = lambda x, x_fs: fantastic_wind_remover(x, x_fs)
# Second case, extra parameters:
#   f = lambda x, x_fs: amazing_wind_remover(x, x_fs, fixed_param1, fixed_param2)

#def benchmark(f, dataset, of):
def benchmark(f, dataset, generate_wav=False):
    """Benchmark method f on a dataset.
    
    f is a function handle to your method, for example:
    Simple case, no extra parameters:
        f = lambda x, x_fs: fantastic_wind_remover(x, x_fs)
    Second case, extra parameters:
        f = lambda x, x_fs: amazing_wind_remover(x, x_fs, fixed_param1, fixed_param2)
    
    dataset is a number between 0 and 7:
        0: wind1, SNR =   0dB
        1: wind1, SNR =  -5dB
        2: wind1, SNR = -10dB
        3: wind1, SNR = -20dB
        4: wind2, SNR =   0dB
        5: wind2, SNR =  -5dB
        6: wind2, SNR = -10dB
        7: wind2, SNR = -20dB
    If generate_wav is True, wav files will be generated, with both the input and the results.
    
    Each dataset is comprised of 6 speech samples, 3 males and 3 females, all corrupted with the same wind.
    
    This function outputs 5 things:
        q: method gain (dB) (q_output - q_input)
        y: list with the results for each speech sample
        y_fs: sampling rate of the results
        q_input: input signal SNR (compared to the ground truth)
        q_output: result SNR (compared to the ground truth)
    
    Just print(q) to show how good your method is. You can mostly ignore the other outputs.
    
    Check run_benchmark for an example.
    """

    of = lambda x_true, x: compute_snr(x_true, x)
    
    wind_f_stem = os.path.join('data', 'test_raw', 'wind')
    voice_f_stem = os.path.join('data', 'test_raw')
    
    out_stem = 'benchmark_'
    
    voice_f = ('female1', 'female2', 'female3', 'male1', 'male2', 'male3')
    
    if dataset == 0:
        snr_in = 0
        wind_f = 'wind1'
    elif dataset == 1:
        snr_in = -5
        wind_f = 'wind1'
    elif dataset == 2:
        snr_in = -10
        wind_f = 'wind1'
    elif dataset == 3:
        snr_in = -20
        wind_f = 'wind1'
    elif dataset == 4:
        snr_in = 0
        wind_f = 'wind2'
    elif dataset == 5:
        snr_in = -5
        wind_f = 'wind2'
    elif dataset == 6:
        snr_in = -10
        wind_f = 'wind2'
    elif dataset == 7:
        snr_in = -20
        wind_f = 'wind2'
        
    fn = len(voice_f)
    
    y = list(range(fn))
    x = list(range(fn))
    x_noisy = list(range(fn))
        
    x_fs = list(range(fn))
    q_input = np.zeros(fn)
    q_output= np.zeros(fn)
    
    for fi in range(fn):
        filename_speech = os.path.join(voice_f_stem, voice_f[fi] + '.wav')
        filename_wind = os.path.join(wind_f_stem, wind_f + '.wav')
        
        x, x_noisy, x_fs[fi] = pre_processing(filename_speech, filename_wind, snr_in)
        y[fi] = f(x_noisy, x_fs[fi])
        q_input[fi] = of(x, x_noisy)
        q_output[fi] = of(x, y[fi])
        
        if generate_wav:
            i_f = out_stem + voice_f[fi] + '_in.wav'
            o_f = out_stem + voice_f[fi] + '_out.wav'
            scipy.io.wavfile.write(i_f, x_fs[fi], x_noisy)
            scipy.io.wavfile.write(o_f, x_fs[fi], y[fi])
    
    q = q_output - q_input
    y_fs = x_fs
    
    return (q, y, y_fs, q_input, q_output)
