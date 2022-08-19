import torch
import time
import shutil
import towhee
import towhee.compiler
from unittest import TestCase

timm_model_name = ['resnet50', 'resnet101']
transformers_model_name = ['distilbert-base-cased']

cached_dir = 'test_hub'
towhee.compiler.config.cached_dir = cached_dir


def timm_ori(model_name):
    return (towhee.dummy_input()
            .image_embedding.timm(model_name=model_name)
            .as_function()
            )


def timm_jit(model_name):
    return (towhee.dummy_input()
            .set_jit('towhee')
            .image_embedding.timm(model_name=model_name)
            .as_function()
            )


def transformers_ori(model_name):
    return (towhee.dummy_input()
            .text_embedding.transformers(model_name=model_name)
            .as_function()
            )


def transformers_jit(model_name):
    return (towhee.dummy_input()
            .set_jit('towhee')
            .text_embedding.transformers(model_name=model_name)
            .as_function()
            )


class TestHub(TestCase):
    def test_timm(self):
        data = towhee.ops.image_decode()('towhee_logo.png')
        for name in timm_model_name:
            timm_func0 = timm_ori(name)
            _ = timm_func0(data)
            t1 = time.time()
            res0 = timm_func0(data)
            t2 = time.time()

            timm_func1 = timm_jit(name)
            _ = timm_func1(data)
            t3 = time.time()
            res1 = timm_func1(data)
            t4 = time.time()

            print(f'timm_model: {name}: {t4 - t3} vs {t2 - t1}, {res0} \n{res1}')
            self.assertTrue(t2 - t1 > t4 - t3)

    def test_bert(self):
        data = 'hello world'
        for name in transformers_model_name:
            timm_func0 = transformers_ori(name)
            _ = timm_func0(data)
            t1 = time.time()
            res0 = timm_func0(data)
            t2 = time.time()

            timm_func1 = transformers_jit(name)
            _ = timm_func1(data)
            t3 = time.time()
            res1 = timm_func1(data)
            t4 = time.time()

            print(f'transformer_model: {name} {t4 - t3} vs {t2 - t1}, {res0} \n{res1}')
            self.assertTrue(t2 - t1 > t4 - t3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cached_dir)
