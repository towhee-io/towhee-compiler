import weakref
from types import FrameType

from torchdynamo.bytecode_transformation import is_generator
from torchdynamo.exc import Unsupported
from torchdynamo.guards import GuardedCode


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

    def check_cache(self, frame: FrameType, cache_size: int) -> bool:
        FrameCompiler.input_codes.add(frame.f_code)
        if frame.f_code in FrameCompiler.output_codes:
            return False
        if frame.f_code.co_name == "__setattr__":
            return False
        if (
            frame.f_code.co_name == "<module>"
            and frame.f_code.co_filename == "<string>"
        ):
            return False
        if (
            frame.f_code.co_name == "<lambda>"
            and frame.f_code.co_filename == "<string>"
            and not bool(frame.f_builtins)
        ):
            return False
        if is_generator(frame.f_code):
            raise Unsupported("generator is not supported by towhee.compiler")
        return True
