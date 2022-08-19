import time
import torch
import torchvision.models as models
import numpy as np
import towhee.compiler
from towhee.compiler import jit_compile
from towhee.compiler import set_log_level, get_logger

set_log_level("info")

log = get_logger(__name__)

towhee.compiler.config.debug = True
# towhee.compiler.config.trace = True

torch_model = models.resnet50()
torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
torch_model = torch_model.eval()


def image_embedding0(imgs):
    imgs = torch.tensor(imgs)
    embedding = torch_model(imgs).detach().numpy()
    return embedding.reshape([2048])


def image_embedding1(imgs):
    with jit_compile("nebullvm", feature=True):
        return image_embedding0(imgs)


def test_cnn():
    inputs1 = np.random.randn(1, 3, 244, 244).astype(np.float32)

    print("==== 1 ====")
    t1 = time.time()
    res1 = image_embedding1(inputs1)
    print("1:", time.time() - t1, str(res1)[:50])

    print("==== 2 ====")
    t1 = time.time()
    res2 = image_embedding1(inputs1)
    print("2:", time.time() - t1, str(res2)[:50])

    print("==== 3 ====")
    t1 = time.time()
    res3 = image_embedding0(inputs1)
    print("ori:", time.time() - t1, str(res3)[:50])


test_cnn()
