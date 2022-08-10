import torch
import numpy
import towhee
import time
from towhee.compiler import jit_compile
import torchdynamo

# torchdynamo.config.debug = True

@towhee.register(name='inner_distance')
def inner_distance(query, data):
    dists = []
    for vec in data:
        dist = 0
        for i in range(len(vec)):
            dist += vec[i] * query[i]
        dists.append(dist)
    return dists


data = [numpy.random.random((10000, 128)) for _ in range(10)]
query = numpy.random.random(128)

t1 = time.time()
dc1 = (
    towhee.dc['a'](data)
    .runas_op['a', 'b'](func=lambda _: query)
    .inner_distance[('b', 'a'), 'c']()
)
t2 = time.time()
dc2 = (
    towhee.dc['a'](data)
    .runas_op['a', 'b'](func=lambda _: query)
    .set_jit('towhee')
    .inner_distance[('b', 'a'), 'c']()
)
t3 = time.time()
print('time:', t3-t2, t2-t1, '\n', dc1[0].c[0:2], '\n', dc2[0].c[0:2])
assert(t3-t2 < t2-t1)
