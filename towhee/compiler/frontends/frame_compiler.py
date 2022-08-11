from types import FrameType
import weakref
from towhee.compiler.bytecode.numba_codegen import numba_codegen

import torchdynamo
from torchdynamo.guards import GuardedCode

__NUMBA_CODE_CACHE__ = {}


def numba_compile(frame):
    import numba

    frame_name = frame.f_code.co_name
    print('numba compile:', frame_name)
    if frame_name in ['__exit__', 'nothing', '<lambda>']:
        return None

    def tmp_func(): pass
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


class _Tracker(dict):
    def add(self, strong_obj):
        idx = id(strong_obj)
        if id(strong_obj) not in self:
            self[idx] = weakref.ref(strong_obj, lambda _: self.pop(idx))

    def __contains__(self, item):
        return dict.__contains__(self, id(item))

    @property
    def seen(self):
        return list(self.values())


class FrameCompiler:
    input_codes = _Tracker()
    output_codes = _Tracker()

    def __init__(self) -> None:
        pass

    def __call__(self, frame: FrameType, cache_size: int) -> GuardedCode:
        pass
