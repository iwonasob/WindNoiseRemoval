# -*- coding: utf-8 -*-

from WienerWindFilter import WienerWindFilter
from benchmark import benchmark

wwf = WienerWindFilter()

f = lambda x, x_fs: wwf.apply(x)
q = benchmark(f, 0, True)[0]

print(q)