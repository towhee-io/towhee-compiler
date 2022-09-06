# Towhee Compiler

Towhee compiler is a Python JIT compiler that speeds up AI-related codes by native code generation. The project is inspired by [Numba](https://github.com/numba/numba), [Pyjion](https://www.trypyjion.com) and [TorchDynamo](https://github.com/pytorch/torchdynamo). Towhee compiler uses a frame evaluation hook (see [PEP 523]: https://www.python.org/dev/peps/pep-0523/) to get the chance of compiling python bytecodes into native code.

> The code is based on a forked version of torchdynamo, which extract `fx.Graph` by trace the execution of python code. But the goal of towhee compiler is `whole program code generation`, which also includes program that can not be represented by `fx.Graph`.

## Install with source code

```bash
$ git clone -b nebullvm https://github.com/towhee-io/towhee-compiler.git
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
