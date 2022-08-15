import threading
from typing import Callable

from towhee.compiler.jit import _eval_frame as _C

from torchdynamo import skipfiles
from torchdynamo.exc import BackendCompilerFailed

from ..frontends.compiler_dispatcher import CompilerDispatcher
from ..log import get_logger

log = get_logger(__name__)

safe_compile_lock = threading.Lock()


def _make_safe_frame_compile_fn(graph_compile_fn: Callable, guard_export_fn=None):
    # frame_compile_fn = convert_frame_assert(graph_compile_fn, guard_export_fn)
    frame_compile_fn = CompilerDispatcher(graph_compile_fn, one_graph=True)

    def safe_compile_fn(frame, cache_size):
        try:
            if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
                log.debug(f"skipping {frame.f_code.co_name} {frame.f_code.co_filename}")
                return None
            if (
                frame.f_code.co_filename == "<string>"
                and frame.f_code.co_name == "__new__"
            ):
                # nametuple constructor
                return None
            with safe_compile_lock:
                return frame_compile_fn(frame, cache_size)
        except BackendCompilerFailed as exc:
            log.warn(
                f"Error while processing frame {frame.f_code.co_name}@{frame.f_code.co_filename}:"
                f"Exception stack: {exc}",
            )
            return None
        except Exception as exc:
            log.warn(
                f"Error while processing frame {frame.f_code.co_name}@{frame.f_code.co_filename}:",
            )
            log.warn(f"{exc}")
            return None

    return safe_compile_fn


class CompilerContext:
    def __init__(self, compile_fn, patch_fn=None, extra_ctx=None):
        super().__init__()
        self.compile_fn = compile_fn
        self.patch_fn = patch_fn

        self.prior = None
        self.extra_ctx = extra_ctx
        # log.info(f"==== using new towhee compile decorator ====")

    def __enter__(self):
        if self.extra_ctx:
            self.extra_ctx.__enter__()
        self.prior = _C.set_eval_frame(self.compile_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _C.set_eval_frame(self.prior)
        self.prior = None
        if self.extra_ctx:
            self.extra_ctx.__exit__(exc_type, exc_val, exc_tb)

    def wrap(self, fn):
        def _fn(*args, **kwargs):
            prior = _C.set_eval_frame(self.compile_fn)
            retval = fn(*args, **kwargs)
            _C.set_eval_frame(prior)
            return retval

        return _fn


def compile(backend):
    safe_frame_compile_fn = _make_safe_frame_compile_fn(backend)
    return CompilerContext(safe_frame_compile_fn)
