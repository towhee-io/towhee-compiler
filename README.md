# Towhee Compiler

Towhee compiler is a Python JIT compiler that speeds up AI-related codes by native code generation. The project is inspired by [Numba](https://github.com/numba/numba), [Pyjion](https://www.trypyjion.com) and [TorchDynamo](). Towhee compiler uses a frame evaluation hook (see [PEP 523]: https://www.python.org/dev/peps/pep-0523/) to get the chance of compiling python bytecodes into native code.

The code is based on a forked version of torchdynamo, which extract `fx.Graph` by trace the execution of python code. But the goal of towhee compiler is `whole program code generation`, which also includes program that can not be represented by `fx.Graph`.