import dis
import sys

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_ABSOLUTE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
    # TODO(jansel): double check exception handling
}
if sys.version_info >= (3, 9):
    TERMINAL_OPCODES.add(dis.opmap["RERAISE"])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)

if sys.version_info < (3, 8):

    def stack_effect(opcode, arg, jump=None):
        # jump= was added in python 3.8, we just ingore it here
        if dis.opname[opcode] in ("NOP", "EXTENDED_ARG"):
            # for some reason NOP isn't supported in python 3.7
            return 0
        return dis.stack_effect(opcode, arg)


else:
    stack_effect = dis.stack_effect

from . import bytecode
from .pass_manager import PassManager

__all__ = [
    "bytecode",
    "PassManager",
]
