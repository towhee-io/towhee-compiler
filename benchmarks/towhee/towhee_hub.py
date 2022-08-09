import time
import torch
import nebullvm
import towhee

timm_model_name = ['vgg16', 'vgg19', 'resnet50', 'resnet101']


def test_timm(model_name):
    (towhee.dc(['towhee_logo.png'])
     .image_decode()
     .set_jit('towhee')
     .image_embedding.timm(model_name=model_name)
     )


def test_hub():
    for name in timm_model_name:
        test_timm(name)


test_hub()
