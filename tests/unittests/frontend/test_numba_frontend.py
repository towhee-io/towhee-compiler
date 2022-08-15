import logging
import time
from unittest import TestCase

import numpy as np
from towhee.compiler import jit_compile

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def inner_distance(query, data):
    dists = []
    for vec in data:
        dist = 0
        for i in range(len(vec)):
            dist += vec[i] * query[i]
        dists.append(dist)
    return dists


def inner_distance_kwargs(query, data, bias=0.0):
    dists = []
    for vec in data:
        dist = bias
        for i in range(len(vec)):
            dist += vec[i] * query[i]
        dists.append(dist)
    return dists


class test:
    @staticmethod
    def inner_distance(query, data):
        dists = []
        for vec in data:
            dist = 0
            for i in range(len(vec)):
                dist += vec[i] * query[i]
            dists.append(dist)
        return dists

    def my_inner_distance(self, query, data):
        dists = []
        for vec in data:
            dist = 0
            for i in range(len(vec)):
                dist += vec[i] * query[i]
            dists.append(dist)
        return dists


class TestNumbaFrontend(TestCase):
    # DONE: add unit test for function with kwargs
    # DONE: add unit test for function with args and kwargs
    # TODO: add unit test for instance method
    # TODO: add unit test for class method
    # DONE: add unit test for static method
    def setUp(self) -> None:
        self.data = [np.random.random((10000, 64)) for _ in range(10)]
        self.query = np.random.random(64)

        self.x1 = np.array([1, 2, 3])
        self.x2 = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

    def test_numba_compile(self):
        with jit_compile(feature=True):
            result = inner_distance(self.x1, self.x2)
        self.assertListEqual(result, [14, 32])

    def test_numba_compile_with_kwargs(self):
        with jit_compile(feature=True):
            result = inner_distance_kwargs(self.x1, self.x2, bias=1.0)
        self.assertListEqual(result, [15.0, 33.0])

    def test_numba_compile_static_method(self):
        with jit_compile(feature=True):
            result = test.inner_distance(self.x1, self.x2)
        self.assertListEqual(result, [14, 32])

    # TODO: fix the test case for instance method
    # def test_numba_compile_instance_method(self):
    #     with jit_compile(feature=True):
    #         t = test()
    #         result = t.my_inner_distance(self.x1, self.x2)
    #     self.assertListEqual(result, [14, 32])

    def test_numba_benchmark(self):
        t1 = time.time()
        [inner_distance(self.query, d) for d in self.data]
        t2 = time.time()
        with jit_compile(feature=True):
            for d in self.data:
                inner_distance(self.query, d)
        t3 = time.time()
        logging.info(f"{t3 - t2} vs {t2 - t1}")
        self.assertLess(t3 - t2, t2 - t1)
