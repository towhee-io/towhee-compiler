import hashlib
from types import FrameType
from typing import Callable

import torchdynamo
from towhee.compiler.bytecode.numba_codegen import numba_codegen

from torchdynamo.guards import GuardedCode

from .frame_compiler import FrameCompiler
from .torch_frame_compiler import _try_resolve_compiler_fn


__TORCH_CODE_CACHE__ = {}


def torch_compile(frame: FrameType, graph_compile_fn: Callable):
    from torch import fx

    cache_target = id(frame.f_code)

    def tmp_func():
        pass

    # import ipdb; ipdb.set_trace()
    tmp_func.__code__ = frame.f_code

    try:
        import torch
        import torch.onnx

        global __TORCH_CODE_CACHE__
        if (
            frame.f_code.co_name == "forward"
            and frame.f_code.co_varnames[0] == "self"
            and "self" in frame.f_locals
            and isinstance(frame.f_locals["self"], (torch.nn.Module,))
        ):
            args = []
            for i in range(1, frame.f_code.co_argcount):
                name = frame.f_code.co_varnames[i]
                args.append(frame.f_locals[name])
            model = frame.f_locals["self"]
            args = tuple(args)
            compiled_fn = graph_compile_fn(model, example_inputs=args)
            def wrapper(*wargs, **wkwargs):
                return compiled_fn(*wargs[1:], **wkwargs)[0]
            __TORCH_CODE_CACHE__[cache_target] = torchdynamo.disable(wrapper)
            frame.f_globals["__TORCH_CODE_CACHE__"] = __TORCH_CODE_CACHE__
            retval = GuardedCode(
                check_fn=None,
                code=numba_codegen(frame, cache_target, "__TORCH_CODE_CACHE__"),
            )
            return retval
    except:
        import traceback

        traceback.print_exc()
        import ipdb

        ipdb.set_trace()
        return None


class TorchFrameCompilerV2(FrameCompiler):
    def __init__(self, graph_compile_fn: Callable) -> None:
        super().__init__()
        self.graph_compile_fn = _try_resolve_compiler_fn(graph_compile_fn)

    def call(self, frame: FrameType) -> GuardedCode:
        try:
            retval = torch_compile(frame, self.graph_compile_fn)
            print(f"done {frame.f_code.co_filename}:{frame.f_code.co_firstlineno}")
            return retval
        except:
            print(f"failed {frame.f_code.co_filename}:{frame.f_code.co_firstlineno}")
