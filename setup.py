#!/usr/bin/env python
from pathlib import Path

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension

HERE = Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name="towhee.compiler",
    version="0.1.1",
    url="https://github.com/towhee-io/towhee-compiler",
    description="A JIT compiler for accelerating AI programs written in python.",
    long_description=README,
    long_description_content_type="text/markdown",
    author='Towhee Team',
    author_email='towhee-team@zilliz.com',
    license="BSD-3",
    keywords="pytorch machine learning compilers",
    python_requires=">=3.7, <3.11",
    install_requires=["nebullvm<=0.4.0", "numpy", "recordclass", "tabulate",
                      "torch>=1.12.0", "torchvision", "typeguard"],
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
