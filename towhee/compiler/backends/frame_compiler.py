from types import FrameType
import weakref
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
