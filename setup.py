#!/usr/bin/env python
from typing import List

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension

long_description = """
Towhee compiler is a Python JIT compiler that
speeds up AI-related codes by native code generation.
The project is inspired by [Numba](https://github.com/numba/numba),
[Pyjion](https://www.trypyjion.com) and [TorchDynamo](https://github.com/pytorch/torchdynamo).
Towhee compiler uses a frame evaluation hook (see [PEP 523]: https://www.python.org/dev/peps/pep-0523/)
to get the chance of compiling python bytecodes into native code.
"""

def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]

setup(
    name="towhee.compiler",
    version="0.1.0",
    url="https://github.com/towhee-io/towhee-compiler",
    description="A JIT compiler for accelerating AI programs written in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Towhee Team',
    author_email='towhee-team@zilliz.com',
    license="BSD-3",
    keywords="pytorch machine learning compilers",
    python_requires=">=3.7, <3.11",
    install_requires=parse_requirements('requirements.txt'),
    packages=find_packages(
        include=[
            "towhee",
            "towhee.compiler*",
            "torchdynamo*",
        ]
    ),
    namespace_package=["towhee"],
    zip_safe=False,
    ext_modules=[
        Extension(
            "towhee.compiler.jit._eval_frame",
            ["towhee/compiler/jit/_eval_frame.c"],
            extra_compile_args=["-Wall"],
        ),
        CppExtension(
            name="torchdynamo._guards",
            sources=["torchdynamo/_guards.cpp"],
            extra_compile_args=["-std=c++14"],
        ),
    ],
)
