import shutil
import time
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch
import torchvision.models as models
import towhee
import towhee.compiler
from towhee.compiler import jit_compile

# towhee.compiler.config.debug = True

cached_dir = "test_nnop"
towhee.compiler.config.cached_dir = cached_dir


@towhee.register
class image_embedding0:
    def __init__(self):
        torch_model = models.resnet50()
        torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
        _ = torch_model.eval()
        self.torch_model = torch_model

    def __call__(self, imgs):
        imgs = torch.tensor(imgs)
        embedding = self.torch_model(imgs)
        return embedding


@towhee.register
class image_embedding1:
    def __init__(self):
        torch_model = models.resnet50()
        torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
        _ = torch_model.eval()
        self.torch_model = torch_model

    def __call__(self, imgs):
        imgs = torch.tensor(imgs)
        embedding = self.torch_model(imgs)
        return embedding


class TestNNop(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = [
            np.random.randn(1, 3, 244, 244).astype(np.float32) for _ in range(5)
        ]
        op1 = towhee.ops.image_embedding0()
        _ = op1(self.data[0])
        t1 = time.time()
        _ = op1(self.data[0])
        t2 = time.time()
        self.ori_time = t2 - t1

    def test_set_jit(self):
        res0 = towhee.dc["a"](self.data).set_jit("towhee").image_embedding0["a", "b"]()
        self.assertTrue(Path(cached_dir).is_dir())
        print(f"set_jit result: {res0[0].b[:10]}")

    def test_nebullvm(self):
        op3 = towhee.ops.image_embedding0()
        _ = op3(self.data[0])
        with jit_compile(backend="nebullvm"):
            _ = op3(self.data[0])
            t1 = time.time()
            res1 = op3(self.data[0])
            t2 = time.time()
        print(f"nebullvm: {t2 - t1} vs {self.ori_time}, {res1}")
        self.assertTrue(t2 - t1 < self.ori_time)

    def test_ofi(self):
        op2 = towhee.ops.image_embedding1()
        _ = op2(self.data[0])
        with jit_compile(backend="ofi"):
            _ = op2(self.data[0])
            t1 = time.time()
            res2 = op2(self.data[0])
            t2 = time.time()
        print(f"ofi: {t2 - t1} vs {self.ori_time}, {res2}")
        self.assertTrue(t2 - t1 < self.ori_time)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cached_dir)
