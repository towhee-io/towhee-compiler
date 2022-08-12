from types import FrameType

from towhee.compiler.bytecode.numba_codegen import numba_codegen

import torchdynamo
from torchdynamo.guards import GuardedCode

from .frame_compiler import FrameCompiler

__NUMBA_CODE_CACHE__ = {}


def numba_compile(frame: FrameType):
    import numba

    frame_name = frame.f_code.co_name
    if frame_name in ["__exit__", "nothing", "<lambda>"]:
        return None
    else:
        frame_name = ":".join(
            [
                frame_name,
                frame.f_code.co_filename,
                str(frame.f_code.co_firstlineno),
            ]
        )

    def tmp_func():
        pass

    tmp_func.__code__ = frame.f_code
    try:
        global __NUMBA_CODE_CACHE__
        numba_func = numba.njit(tmp_func, cache=True)
        warpper = torchdynamo.disable(numba_func)

        glob_name = "__NUMBA_CODE_CACHE__"
        __NUMBA_CODE_CACHE__[frame_name] = warpper
        frame.f_locals[frame_name] = warpper
        frame.f_globals[glob_name] = __NUMBA_CODE_CACHE__

        numba_func.check_fn = None
        numba_func.code = numba_codegen(frame, frame_name, glob_name)
        return numba_func
    except:
        return None


class NumbaFrameCompiler(FrameCompiler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, frame: FrameType, cache_size: int) -> GuardedCode:
        return numba_compile(frame)
