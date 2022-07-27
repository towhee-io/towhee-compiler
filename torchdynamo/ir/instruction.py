import dis
from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: int
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    @staticmethod
    def from_dis(i: dis.Instruction):
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
        )

    @staticmethod
    def create(name, arg=None, argval=None, target=None):
        return Instruction(
            opcode=dis.opmap[name],
            opname=name,
            arg=arg,
            argval=argval if argval is not None else arg,
            target=target,
        )

def create_instruction(name, arg=None, argval=None, target=None):
    return Instruction.create(name, arg, argval, target)
