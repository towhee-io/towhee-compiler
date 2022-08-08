import time
import torch
import nebullvm
import towhee

timm_model_name = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'efficientnet_b2', 'vit_large_patch16_224']


def test_timm(model_name):
    (towhee.dc(['https://github.com/towhee-io/towhee/blob/main/towhee_logo.png?raw=true'])
     .image_decode()
     .set_jit('towhee')
     .image_embedding.timm(model_name=model_name)
     )


def time_cost(func, *args):
    try:
        t1 = time.time()
        func(*args)
        t2 = time.time()
        return t2-t1
    except:
        print('Failed with func: ', func)


def test_hub():
    for name in timm_model_name:
        c1 = time_cost(test_timm, name)
        c2 = time_cost(test_timm, name)
        assert c1 > c2
        print('time of model:', name, c1, c2)


test_hub()
