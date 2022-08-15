from types import FrameType

from towhee.compiler.bytecode.numba_codegen import numba_codegen

import torchdynamo
from torchdynamo.guards import GuardedCode

from .frame_compiler import FrameCompiler

__NUMBA_CODE_CACHE__ = {}


def numba_compile(frame: FrameType):
    import numba

    if frame.f_code.co_name in ["__exit__", "nothing", "<lambda>"]:
        return None
    cache_target = id(frame.f_code)

    def tmp_func():
        pass

    tmp_func.__code__ = frame.f_code
    try:
        global __NUMBA_CODE_CACHE__
        numba_func = numba.njit(tmp_func, cache=True)
        warpper = torchdynamo.disable(numba_func)

        __NUMBA_CODE_CACHE__[cache_target] = warpper
        frame.f_globals["__NUMBA_CODE_CACHE__"] = __NUMBA_CODE_CACHE__

        numba_func.check_fn = None
        numba_func.code = numba_codegen(frame, cache_target, "__NUMBA_CODE_CACHE__")
        return numba_func
    except:
        return None


class NumbaFrameCompiler(FrameCompiler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, frame: FrameType, cache_size: int = None) -> GuardedCode:
        return numba_compile(frame)
