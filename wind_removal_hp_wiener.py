# -*- coding: utf-8 -*-

import scipy.signal

def wind_removal_hp_wiener(x):
    
    b, a = scipy.signal.butter(3, 200, btype='highpass')
    y_h = scipy.signal.lfilter(b, a, x)
    
    y = scipy.signal.wiener(y_h, 33, 0.01)
        
    return y
