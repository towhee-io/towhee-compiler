import torch
import torchvision.models as models
import numpy as np

import towhee
from towhee import register
from towhee.compiler import jit_compile
import torchdynamo
# import nebullvm

# torchdynamo.config.debug = True
# torchdynamo._torchdynamo_disable=True

@register
class image_embedding:
    def __init__(self):
        torch_model = models.resnet50()
        torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
        _ = torch_model.eval()
        self.torch_model = torch_model

    def __call__(self, imgs):
        imgs = torch.tensor(imgs)
        embedding = self.torch_model(imgs)
        return embedding


@register
class image_embedding_jit:
    def __init__(self):
        torch_model = models.resnet50()
        torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
        _ = torch_model.eval()
        self.torch_model = torch_model

    @torchdynamo.optimize('nebullvm')
    def __call__(self, imgs):
        imgs = torch.tensor(imgs)
        embedding = self.torch_model(imgs).detach().numpy()
        return embedding.reshape([2048])


inputs = [np.random.randn(1, 3, 244, 244).astype(np.float32) for _ in range(1)]


def ori_test():
    vec1 = towhee.dc(inputs).image_embedding()
    print('\n\nvec1:', vec1[0])

def jit_test():
    vec2 = towhee.dc(inputs).image_embedding_jit()
    print('\n\nvec2:', vec2[0])

def set_jit_test():
    vec3 = towhee.dc(inputs).set_jit('towhee').image_embedding()
    print('\n\nvec3:', vec3[0])

def ofi_jit_test():
    op1 = towhee.engine.factory.op('image-embedding')
    with jit_compile('ofi'):
        vec0 = op1(inputs[0])
    print('\n\nvec0:', vec0[0])

def with_jit_test():
    with jit_compile('ofi'):
        vec4 = towhee.dc(inputs).image_embedding()
    print('\n\nvec4:', vec4[0])


# ori_test()
# jit_test()
# set_jit_test()
# ofi_jit_test()
with_jit_test()