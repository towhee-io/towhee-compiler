import logging
from types import FrameType
from typing import Callable

from torchdynamo.guards import GuardedCode

from .frame_compiler import FrameCompiler
from .nn_module_frame_compiler import NNModuleFrameCompiler
from .numba_frame_compiler import NumbaFrameCompiler
from .torch_frame_compiler import TorchFrameCompiler
from .torch_frame_compiler import has_tensor_in_frame


class CompilerDispatcher(FrameCompiler):
    def __init__(self, graph_compile_fn: Callable, one_graph: bool) -> None:
        super().__init__()
        self._numba = NumbaFrameCompiler()
        self._torch = TorchFrameCompiler(graph_compile_fn, one_graph)
        self._nn_module = NNModuleFrameCompiler(graph_compile_fn)

    def __call__(self, frame: FrameType, cache_size: int) -> GuardedCode:
        if not self.check_cache(frame, cache_size):
            return None
        if not has_tensor_in_frame(frame):
            return self._numba(frame, cache_size)
        try:
            compiled = self._nn_module.call(frame)
            if compiled is None:
                raise Exception()
            logging.info("using nn.module compiler")
            return compiled
        except:
            logging.info("failbak to torchdynamo")
            return self._torch.call(frame)
