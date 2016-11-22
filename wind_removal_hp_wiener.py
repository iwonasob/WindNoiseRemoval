# -*- coding: utf-8 -*-

import scipy.signal

def wind_removal_hp_wiener(x, x_fs):
    
    f_nyquist = x_fs / 2
    f_cutoff = 200
    
    b, a = scipy.signal.butter(4,  f_cutoff / f_nyquist, btype='highpass')
    y_h = scipy.signal.lfilter(b, a, x)
    
    
#    y = y_h
    y = scipy.signal.wiener(y_h, 401, 10)
            
    return y
