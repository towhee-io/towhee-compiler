# Towhee Compiler

Towhee compiler is a Python JIT compiler that speeds up AI-related codes by native code generation. The project is inspired by [Numba](https://github.com/numba/numba), [Pyjion](https://www.trypyjion.com) and [TorchDynamo](https://github.com/pytorch/torchdynamo). Towhee compiler uses a frame evaluation hook (see [PEP 523]: https://www.python.org/dev/peps/pep-0523/) to get the chance of compiling python bytecodes into native code.

> The code is based on a forked version of torchdynamo, which extract `fx.Graph` by trace the execution of python code. But the goal of towhee compiler is `whole program code generation`, which also includes program that can not be represented by `fx.Graph`.

## Install

### Install with pip

```bash
$ pip install towhee.compiler==0.1.0
```

### Install with source code

```bash
$ git clone -b 0.1.0 https://github.com/towhee-io/towhee-compiler.git
$ cd towhee-compiler && pip install -r requirements
$ python3 setup.py develop
```

## Examples

### Run with Torch Model

- **Compile**

Towhee compiler can speedup any models, for example, we just need to add `jit_compile` context to the `image_embedding` function.

```python
import torch
import torchvision.models as models
import numpy as np
import towhee.compiler
from towhee.compiler import jit_compile

# towhee.compiler.config.debug = True

torch_model = models.resnet50()
torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1]))
torch_model = torch_model.eval()

def image_embedding(inputs):
    imgs = torch.tensor(inputs)
    embedding = torch_model(imgs).detach().numpy()
    return embedding.reshape([2048])
  
inputs = np.random.randn(1, 3, 244, 244).astype(np.float32)
with jit_compile():
    embeddings = image_embedding(inputs)
```

- **Timer**

We have compiled the model with the nebullvm backend (the default backend in towhee.compiler ), and we can define a Timer class to record the time spent.

```python
import time

class Timer:
    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._interval = time.time() - self._start
        print('%s: %.2fs'%(self._name, self._interval))
```

And we can see that the compiled function is more than 3 times faster.

```python
with Timer('Image Embedding'):
    embeddings = image_embedding(inputs)
    
with Timer('Image Embedding with towhee compiler'), jit_compile():
    embeddings_jit = image_embedding(inputs)
```

Image Embedding: 0.14s

Image Embedding with towhee compiler: 0.04s

### Run with Towhee

Towhee supports setting JIT to use **towhee.compiler** to compile. 

- **Set JIT**

For example, we can add `set_jit('towhee')` in image embedding pipeline, then the following operator will be automatically compiled

```python
import towhee

embeddings_towhee = (
    towhee.dc(['https://raw.githubusercontent.com/towhee-io/towhee/main/towhee_logo.png'])
          .image_decode()
          .set_jit('towhee')
          .image_embedding.timm(model_name='resnet50')
)
```

- **Timer**

And we can make two towhee pipeline function to record the time cost.

```python
towhee_func = (towhee.dummy_input()
            .image_embedding.timm(model_name='resnet50')
            .as_function()
            )

towhee_func_jit = (towhee.dummy_input()
            .set_jit('towhee')
            .image_embedding.timm(model_name='resnet50')
            .as_function()
            )
```

```python
data = towhee.ops.image_decode()('https://raw.githubusercontent.com/towhee-io/towhee/main/towhee_logo.png')

with Timer('Towhee function'):
    emb = towhee_func(data)
    
with Timer('Towhee function with Compiler'):
    emb_jit = towhee_func_jit(data)
```

Towhee function: 0.14s

Towhee function with Compiler: 0.08s

### Tests in Towhee Hub

According to the README of Operator on [Towhee Hub](https://towhee.io/tasks/operator), we set jit to compile and speedup model , theresults are as follows:
> 5.5 means that the performance after jit is 5.5 times, and N means no speedup or compilation failure. And more test results will be updated continuously. 

<table>
   <tr>
      <td><b>Field</b></td>
      <td><b>Task</b></td>
      <td><b>Operator</b></td>
      <td><b>Speedup(CPU/GPU)</b></td>
   </tr>
   <tr>
      <td rowspan="5">Image</td>
      <td rowspan="3">Image Embedding</td>
      <td><a href="https://towhee.io/image-embedding/timm">image_embedding.timm</a></td>
      <td>1.3/1.3</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/image-embedding/data2vec">image_embedding.data2vec</a></td>
      <td>1.2/1.7</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/image-embedding/swag">image_embedding.swag</a></td>
      <td>1.4/N</td>
   </tr>
   <tr>
      <td rowspan="1">Face Embedding</td>
      <td><a href="https://towhee.io/face-embedding/inceptionresnetv1">face_embedding.inceptionresnetv1</a></td>
      <td>3.2/N</td>
   </tr>
   <tr>
      <td rowspan="1">Face Landmark</td>
      <td><a href="https://towhee.io/face-landmark-detection/mobilefacenet">face_landmark_detection.mobilefacenet</a></td>
      <td>2.1/2.1</td>
   </tr>
   <tr>
      <td rowspan="4">NLP</td>
      <td rowspan="4">Text Embedding</td>
      <td><a href="https://towhee.io/text-embedding/transformers">text_embedding.transformers</a></td>
      <td>2.6/N</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/text-embedding/data2vec">text_embedding.data2vec</a></td>
      <td>1.8/N</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/text-embedding/realm">text_embedding.realm</a></td>
      <td>5.5/1.9</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/text-embedding/xlm_prophetnet">text_embedding.xlm_prophetnet</a></td>
      <td>2.1/2.8</td>
   </tr>
   <tr>
      <td rowspan="3">Audio</td>
      <td rowspan="1">Audio Classification</td>
      <td><a href="https://towhee.io/audio-classification/panns">audio_classification.panns</a></td>
      <td>1.6/N</td>
   </tr>
   <tr>
      <td rowspan="2">Audio Embedding</td>
      <td><a href="https://towhee.io/audio-embedding/vggish">audio_embedding.vggish</a></td>
      <td>1.5/N</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/audio-embedding/data2vec">audio_embedding.data2vec</a></td>
      <td>1.5/N</td>
   </tr>
   <tr>
      <td rowspan="3">Multimodal</td>
      <td rowspan="1">Image Text</td>
      <td><a href="https://towhee.io/image-text-embedding/blip">image_text_embedding.blip</a></td>
      <td>2.3/N</td>
   </tr>
   <tr>
      <td rowspan="2">Video Text</td>
      <td><a href="https://towhee.io/video-text-embedding/bridge-former">video_text_embedding.bridge_former(modality='text')</a></td>
      <td>2.1/N</td>
   </tr>
   <tr>
      <td><a href="https://towhee.io/video-text-embedding/frozen-in-time">video_text_embedding.frozen_in_time(modality='text')</a></td>
      <td>2.2/N</td>
   </tr>
</table>







