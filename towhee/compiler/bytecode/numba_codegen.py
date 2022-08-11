import sys
from types import CodeType
from types import FrameType

from torchdynamo.bytecode_transformation import assemble

from . import create_instruction

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


def numba_codegen(frame: FrameType, cache_target: str, cache_name: str):
    instructions = []
    new_code = {k: getattr(frame.f_code, k) for k in keys}
    new_code["co_names"] = new_code["co_names"] + (cache_name,)
    new_code["co_consts"] = new_code["co_consts"] + (cache_target,)
    instructions = [
        create_instruction("LOAD_GLOBAL", len(new_code["co_names"]) - 1),
        create_instruction("LOAD_CONST", len(new_code["co_consts"]) - 1),
        create_instruction("BINARY_SUBSCR"),
    ]
    for i in range(frame.f_code.co_argcount):
        instructions.append(create_instruction("LOAD_FAST", i))
    instructions += [
        create_instruction("CALL_FUNCTION", frame.f_code.co_argcount),
        create_instruction("RETURN_VALUE"),
    ]
    bytecode, lnotab = assemble(instructions, 0)
    if sys.version_info < (3, 10):
        new_code["co_lnotab"] = lnotab
    else:
        new_code["co_linetable"] = lnotab
    new_code["co_code"] = bytecode
    return CodeType(*[new_code[k] for k in keys])

