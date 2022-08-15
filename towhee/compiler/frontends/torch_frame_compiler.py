import dis
import sys
import warnings
from types import CodeType
from types import FrameType
from typing import Callable

import torch
from towhee.compiler import passes

from torchdynamo import config
from torchdynamo.allowed_functions import is_allowed
from torchdynamo.bytecode_transformation import assemble
from torchdynamo.bytecode_transformation import cleaned_instructions
from torchdynamo.bytecode_transformation import devirtualize_jumps
from torchdynamo.bytecode_transformation import fix_extended_args
from torchdynamo.bytecode_transformation import fix_vars
from torchdynamo.bytecode_transformation import stacksize_analysis
from torchdynamo.bytecode_transformation import update_offsets
from torchdynamo.exc import BackendCompilerFailed
from torchdynamo.exc import InternalTorchDynamoError
from torchdynamo.exc import RestartAnalysis
from torchdynamo.exc import SkipFrame
from torchdynamo.exc import TorchRuntimeError
from torchdynamo.exc import Unsupported
from torchdynamo.guards import CheckFunctionManager
from torchdynamo.guards import GuardedCode
from torchdynamo.symbolic_convert import InstructionTranslator
from torchdynamo.utils import CleanupManager
from torchdynamo.utils import ExactWeakKeyDictionary
from torchdynamo.utils import is_namedtuple
from torchdynamo.utils import istype

from .frame_compiler import FrameCompiler

orig_code_map = ExactWeakKeyDictionary()


def debug_print(prefix: str, frame: FrameType):
    if not config.debug:
        return
    print(
        f"\n{prefix}",
        frame.f_code.co_name,
        frame.f_code.co_filename,
        frame.f_code.co_firstlineno,
    )
    print(dis.Bytecode(frame.f_code).dis())


def _try_resolve_compiler_fn(compiler_fn):
    """WrapperBackend if config.verify_correctness is True"""
    if isinstance(compiler_fn, str):
        from towhee.compiler.backends import resolve

        return resolve(compiler_fn)
    return compiler_fn


def has_tensor_in_frame(frame):
    """Check if the frame has torch.* related bits"""
    # Check if the function was decorated with torchdynamo.optimize
    # TODO: re-enable this check
    # if frame.f_code in always_optimize_code_objects:
    #     return True

    # Check if there is global import of torch.*
    for co_name in frame.f_code.co_names:
        if co_name in frame.f_globals:
            if is_allowed(frame.f_globals[co_name]):
                return True

    seen_ids = dict()

    def has_tensor(obj):
        """Recursively check if the obj has a tensor"""
        obj_id = id(obj)
        if obj_id in seen_ids:
            return seen_ids[obj_id]
        seen_ids[obj_id] = False

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif istype(obj, (list, tuple)):
            seen_ids[obj_id] = any([has_tensor(v) for v in obj])
            return seen_ids[obj_id]
        elif istype(obj, dict):
            seen_ids[obj_id] = any([has_tensor(v) for v in obj.values()])
            return seen_ids[obj_id]
        elif istype(obj, (str, int, float, type(None), bool)):
            seen_ids[obj_id] = False
            return seen_ids[obj_id]
        elif is_namedtuple(obj):
            seen_ids[obj_id] = any([has_tensor(getattr(obj, v)) for v in obj._fields])
            return seen_ids[obj_id]
        elif (
            not is_allowed(obj)
            and hasattr(obj, "__dict__")
            and len(getattr(obj, "__dict__"))
        ):
            seen_ids[obj_id] = any([has_tensor(v) for v in obj.__dict__.values()])
            return seen_ids[obj_id]
        else:
            return False

    # Check if the passed arguments are of type Tensor
    for value in frame.f_locals.values():
        if has_tensor(value):
            return True

    return False


class TorchFrameCompiler(FrameCompiler):
    def __init__(self, graph_compile_fn: Callable, one_graph: bool) -> None:
        super().__init__()
        self.graph_compile_fn = _try_resolve_compiler_fn(graph_compile_fn)
        self.one_graph = one_graph

    def __call__(self, frame: FrameType, cache_size: int) -> GuardedCode:
        if not self.check_cache(frame, cache_size):
            return None
        if not has_tensor_in_frame(frame):
            return None
        return self.call(frame)

    def call(self, frame: FrameType) -> GuardedCode:
        try:
            return self.compile(frame)
        except (Unsupported, TorchRuntimeError, BackendCompilerFailed):
            if config.debug or config.trace or config.print_internal_exceptions:
                debug_print("WONT CONVERT", frame)
            raise
        except Exception:
            if config.debug or config.trace or config.print_internal_exceptions:
                import traceback

                debug_print("WONT CONVERT", frame)
                warnings.warn("=" * 10 + " TorchDynamo Stack Trace " + "=" * 10 + "\n")
                traceback.print_exc()
                warnings.warn(
                    "=" * 10 + " Exception (above) while processing " + "=" * 10 + "\n",
                )
                traceback.print_stack(frame)
                warnings.warn("=" * 10 + " End debug info " + "=" * 10 + "\n")
            raise InternalTorchDynamoError()

    def transform(self, frame: FrameType, instructions, code_options):
        tracer = InstructionTranslator(
            instructions,
            frame.f_code,
            frame.f_locals,
            frame.f_globals,
            frame.f_builtins,
            code_options,
            self.graph_compile_fn,
            self.one_graph,
        )
        tracer.run()
        self.output = tracer.output
        instructions[:] = self.output.output_instructions
        code_options.update(self.output.code_options)

        instructions[:] = passes.bytecode.common().execute(instructions)

    def transform_code_object(self, frame: FrameType, safe: bool = False):
        keys = [
            "co_argcount",
            "co_posonlyargcount",  # python 3.8+
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
            "co_firstlineno",
            "co_lnotab",  # changed to "co_linetable" if python 3.10+
            "co_freevars",
            "co_cellvars",
        ]
        if sys.version_info < (3, 8):
            keys.pop(1)
        if sys.version_info >= (3, 10):
            keys = list(map(lambda x: x.replace("co_lnotab", "co_linetable"), keys))
        code_options = {k: getattr(frame.f_code, k) for k in keys}
        assert len(code_options["co_varnames"]) == code_options["co_nlocals"]

        instructions = cleaned_instructions(frame.f_code, safe)

        # transformations(instructions, code_options)
        self.transform(frame, instructions, code_options)

        fix_vars(instructions, code_options)

        dirty = True
        while dirty:
            update_offsets(instructions)
            devirtualize_jumps(instructions)
            # this pass might change offsets, if so we need to try again
            dirty = fix_extended_args(instructions)

        bytecode, lnotab = assemble(instructions, frame.f_code.co_firstlineno)
        if sys.version_info < (3, 10):
            code_options["co_lnotab"] = lnotab
        else:
            code_options["co_linetable"] = lnotab

        code_options["co_code"] = bytecode
        code_options["co_nlocals"] = len(code_options["co_varnames"])
        code_options["co_stacksize"] = stacksize_analysis(instructions)
        assert set(keys) - {"co_posonlyargcount"} == set(code_options.keys()) - {
            "co_posonlyargcount"
        }
        return CodeType(*[code_options[k] for k in keys])

    def compile(self, frame: FrameType) -> GuardedCode:
        for attempt in range(100):
            try:
                code = self.transform_code_object(frame)
                orig_code_map[code] = frame.f_code
                break
            except RestartAnalysis:
                if attempt > 100:
                    raise Unsupported("100+ RestartAnalysis() calls")
            except SkipFrame:
                return None
        if config.debug:
            debug_print("ORIGINAL BYTECODE", frame)
            print("MODIFIED BYTECODE")
            # print(dis.Bytecode(code).info())
            print(dis.Bytecode(code).dis())
        assert self.output.guards is not None
        CleanupManager.instance[code] = self.output.cleanups
        check_fn = CheckFunctionManager(
            self.output.guards, frame.f_locals, frame.f_globals
        )
        guarded_code = GuardedCode(code, check_fn.check_fn)
        if config.debug:
            print("\nGUARDS:")
            for guard in sorted(self.output.guards):
                print(" -", str(guard))
            print()

        return guarded_code
