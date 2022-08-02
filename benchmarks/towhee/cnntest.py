import time

import torch
import torchvision.models as models
from towhee.compiler import jit_compile
import torchdynamo
import numpy as np
import nebullvm
import towhee

torchdynamo.config.debug = True
# torchdynamo.config.trace = True

torch_model = models.resnet50()
torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
_ = torch_model.eval()


@torchdynamo.optimize('nebullvm')
def image_embedding1(imgs):
    imgs = torch.tensor(imgs)
#     imgs = torch.unsqueeze(imgs, 0)
    embedding = torch_model(imgs).detach().numpy()
    return embedding.reshape([2048])



def test_cnn():
    inputs1 = np.random.randn(1, 3, 244, 244).astype(np.float32)
    inputs2 = np.random.randn(1, 3, 244, 244).astype(np.float32)

    t1 = time.time()
    res1 = image_embedding1(inputs1)
    print('1:', time.time()-t1, res1[0])

    t1 = time.time()
    res2 = image_embedding1(inputs1)
    print('2:', time.time()-t1, res2[0])

    t1 = time.time()
    embedding = torch_model(torch.tensor(inputs1)).detach().numpy()
    res3 = embedding.reshape([2048])
    print('ori:', time.time()-t1, res3[0])


test_cnn()


# import urllib
# from PIL import Image
# import timm
# from timm.data.transforms_factory import create_transform
# from timm.data import resolve_data_config
#
#
# class timm_embedding:
#     def __init__(self, model_name):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model = timm.create_model(model_name, pretrained=True)
#         model.eval()
#         model.to(self.device)
#         self.model = model
#
#         url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#         urllib.request.urlretrieve(url, filename)
#         img = Image.open(filename).convert('RGB')
#
#         config = resolve_data_config({}, model=model)
#         transform = create_transform(**config)
#         x = transform(img).unsqueeze(0)
#         self.x = x.to(self.device)
#
#     def __call__(self):
#         out = self.model(self.x)
#         vecs = torch.nn.functional.softmax(out[0], dim=0)
#         return vecs
#
#
# class timm_embedding_ofi:
#     def __init__(self, model_name):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model = timm.create_model(model_name, pretrained=True)
#         model.eval()
#         model.to(self.device)
#         self.model = model
#
#         url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#         urllib.request.urlretrieve(url, filename)
#         img = Image.open(filename).convert('RGB')
#
#         config = resolve_data_config({}, model=model)
#         transform = create_transform(**config)
#         x = transform(img).unsqueeze(0)
#         self.x = x.to(self.device)
#
#     @torchdynamo.optimize('ofi')
#     def __call__(self):
#         out = self.model(self.x)
#         vecs = torch.nn.functional.softmax(out[0], dim=0)
#         return vecs
#
#
# class timm_embedding_nb:
#     def __init__(self, model_name):
#         model = timm.create_model(model_name, pretrained=True)
#         model.eval()
#         self.model = model
#
#         url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#         urllib.request.urlretrieve(url, filename)
#         img = Image.open(filename).convert('RGB')
#
#         config = resolve_data_config({}, model=model)
#         transform = create_transform(**config)
#         x = transform(img).unsqueeze(0)
#         self.x = x
#
#     @torchdynamo.optimize('nebullvm')
#     def __call__(self):
#         out = self.model(self.x)
#         vecs = torch.nn.functional.softmax(out[0], dim=0)
#         return vecs
#
#
# model_name = 'resnet101'
# timm_model = timm_embedding(model_name)
# nb_model = timm_embedding_nb(model_name)
# ofi_model = timm_embedding_ofi(model_name)
#
#
# t1 = time.time()
# vec1 = timm_model()
# print('\ntimm:', time.time()-t1)
# t1 = time.time()
# vec1 = timm_model()
# print(time.time()-t1, vec1[0:3])
#
# t1 = time.time()
# vec2 = ofi_model()
# print('\nofi', time.time()-t1)
# t1 = time.time()
# vec2 = ofi_model()
# print(time.time()-t1, vec2[0:3])
#
# t1 = time.time()
# vec3 = nb_model()
# print('\nnb:', time.time()-t1)
# t1 = time.time()
# vec3 = nb_model()
# print(time.time()-t1, vec3[0:3])
