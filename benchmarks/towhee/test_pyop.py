import torch
import time
import numpy as np
from unittest import TestCase

import towhee

@towhee.register(name='inner_distance')
def inner_distance(query, data):
    dists = []
    for vec in data:
        dist = 0
        for i in range(len(vec)):
            dist += vec[i] * query[i]
        dists.append(dist)
    return dists


class TestPyop(TestCase):
    def setUp(self):
        self.data = [np.random.random((10000, 128)) for _ in range(10)]
        self.query = np.random.random(128)

        t1 = time.time()
        self.ori_res = (
            towhee.dc['a'](self.data)
            .runas_op['a', 'b'](func=lambda _: self.query)
            .inner_distance[('b', 'a'), 'c']()
        )
        t2 = time.time()
        self.ori_time = t2 - t1

    def test_set_jit(self):
        t1 = time.time()
        res = (
            towhee.dc['a'](self.data)
            .runas_op['a', 'b'](func=lambda _: self.query)
            .set_jit('towhee')
            .inner_distance[('b', 'a'), 'c']()
        )
        t2 = time.time()
        print(f'ofi: {t2 - t1} vs {self.ori_time}, {self.ori_res[0].c[:5]} \n{res[0].c[:5]}')
        self.assertTrue(t2-t1 < self.ori_time)
