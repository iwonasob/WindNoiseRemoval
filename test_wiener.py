# -*- coding: utf-8 -*-

from WienerWindFilter import WienerWindFilter
from benchmark import benchmark

wwf = WienerWindFilter(0, 0, 1024)

f = lambda x, x_fs: wwf.apply(x, 128)
b = benchmark(f, 0, True)

print(b[0])