from contextlib import contextmanager
from itertools import chain
from threading import local

import sympy
from torch.fx.graph import inplace_methods
from torch.fx.graph import magic_methods

import torchdynamo

threadlocal = local()


class Virtualized:
    """
    A global variable that redirects via thread local variable

    This allows us to swap in different op implementations in codegen.
    """

    def __init__(self, vname, default):
        self._key = f"__torchinductor_{vname}"
        self._default = default

    def _set_handler(self, value):
        prior = self._get_handler()
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            yield
            self._set_handler(prior)

        return ctx()

    def _get_handler(self):
        try:
            return getattr(threadlocal, self._key)
        except AttributeError:
            return self._default()

    def __getattr__(self, name):
        return getattr(self._get_handler(), name)


class NullHandler:
    pass


class MockHandler:
    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = list(map(str, args))
            fargs.extend(f"{k}={v}" for k, v in kwargs.items())
            return f"{name}({', '.join(fargs)})"

        return inner

    @staticmethod
    def masked(mask, body, other):
        return f"masked({mask}, {body()}, {other})"

    @staticmethod
    def indirect_indexing(index_var):
        return sympy.Symbol(str(index_var))

    @classmethod
    def _init_cls(cls):
        def make_handler(format_string):
            @staticmethod
            def inner(*args):
                return format_string.format(*args)

            return inner

        for name, format_string in chain(
            magic_methods.items(), inplace_methods.items()
        ):
            setattr(cls, name, make_handler(format_string))


class WrapperHandler:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)


MockHandler._init_cls()

ops = Virtualized("ops", MockHandler)
_graph = Virtualized("graph", NullHandler)
_kernel = Virtualized("kernel", NullHandler)


class _V:
    MockHandler = MockHandler
    WrapperHandler = WrapperHandler

    set_ops_handler = ops._set_handler
    get_ops_handler = ops._get_handler
    set_graph_handler = _graph._set_handler
    set_kernel_handler = _kernel._set_handler

    @property
    def ops(self) -> MockHandler:
        """The operator handler specific to the current codegen task"""
        return ops._get_handler()

    @property
    def graph(self) -> "torchdynamo.graph.Graph":
        """The graph currently being generated"""
        return _graph._get_handler()

    @property
    def kernel(self) -> "torchdynamo.codegen.common.Kernel":
        """The kernel currently being generated"""
        return _kernel._get_handler()


V = _V()
