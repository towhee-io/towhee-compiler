import threading
import warnings

from towhee.compiler.jit import _eval_frame as _C

from torchdynamo import config
from torchdynamo import convert_frame
from torchdynamo import skipfiles

safe_compile_lock = threading.Lock()


def _make_safe_compile_fn(compile_fn):
    def safe_compile_fn(frame, cache_size):
        try:
            if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
                if config.debug:
                    print(f"skipping {frame.f_code.co_name} {frame.f_code.co_filename}")
                return None
            if (
                frame.f_code.co_filename == "<string>"
                and frame.f_code.co_name == "__new__"
            ):
                # nametuple constructor
                return None
            with safe_compile_lock:
                return compile_fn(frame, cache_size)
        except Exception:
            warnings.warn(
                "default",
                f"Error while processing frame {frame.f_code.co_name}@{frame.f_code.co_filename}:",
            )
            return None

    return safe_compile_fn


class CompilerContext:
    def __init__(self, compile_fn, patch_fn=None, extra_ctx=None):
        super().__init__()
        self.compile_fn = compile_fn
        self.patch_fn = patch_fn

        self.prior = None
        self.extra_ctx = extra_ctx
        print(f"==== using new towhee compile decorator ====")

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
    compile_fn = convert_frame.convert_frame(backend)
    safe_compile_fn = _make_safe_compile_fn(compile_fn)
    return CompilerContext(safe_compile_fn)
