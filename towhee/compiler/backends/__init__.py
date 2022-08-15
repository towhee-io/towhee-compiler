from typing import Callable

from .backend_compiler import BackendCompiler
from .nebullvm_compiler import NebullvmCompiler


def resolve(name: str) -> Callable:
    if name in BackendCompiler.backends:
        return BackendCompiler.backends[name]()

    from torchdynamo.optimizations.backends import BACKENDS

    return BACKENDS[name]


__all__ = [
    "BackendCompiler",
    "NebullvmCompiler",
    "resolve",
]
