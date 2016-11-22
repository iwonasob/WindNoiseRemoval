# -*- coding: utf-8 -*-

import numpy as np

def compute_snr(x_true, x):
    noise = x - x_true
    
    q = 2 * 10 * np.log10(np.linalg.norm(x) / np.linalg.norm(noise))
    
    return q
