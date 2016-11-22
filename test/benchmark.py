# -*- coding: utf-8 -*-

import numpy as np

def benchmark(f, dataset, of):
    
    [x, x_noisy, x_fs] = load_dataset(dataset);
    
    dn = len(x);
    
    y = list(dn)
    q_input = np.zeros(dn)
    q_output= np.zeros(dn)
    
    for di in range(dn):
        y[di] = f(x_noisy[di])
        q_input[di] = of(x[di], x_noisy[di])
        q_output[di] = of(x[di], y[di])
    
    q = q_input - q_output
    y_fs = x_fs
    
    return (q, y, y_fs, q_input, q_output)