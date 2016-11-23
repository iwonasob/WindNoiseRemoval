# -*- coding: utf-8 -*-

from sparse_nmf import sparse_nmf
from benchmark import benchmark

f = lambda x, x_fs: sparse_nmf(x, x_fs)
b = benchmark(f, 0, True)

print(b[0])