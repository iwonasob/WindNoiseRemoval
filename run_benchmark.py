# -*- coding: utf-8 -*-

#import scipy.io.wavfile

from benchmark import benchmark
from wind_removal_hp_wiener import wind_removal_hp_wiener

f = lambda x, x_fs: wind_removal_hp_wiener(x, x_fs)
#f = lambda x, x_fs: x

q, y, y_fs, q_input, q_output = benchmark(f, 0, True)

print(q)
print(q_input)
print(q_output)
