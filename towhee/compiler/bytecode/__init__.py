import dis
from typing import Any, Optional

from recordclass import dataobject


# @dataclass
class Instruction(dataobject):
    """Mutable Instruction with acceleration and storage optimization

    Examples:

    1. create an instruction:

    >>> inst = Instruction(0, "test_inst", None, 0)
    >>> inst
    Instruction(opcode=0, opname='test_inst', arg=None, argval=0, offset=None, starts_line=None, is_jump_target=False, target=None, argrepr=None)

    2. the instruction object is mutable:

    >>> inst.opcode = 1
    """

    opcode: int
    opname: str
    arg: int
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Optional[dis.Instruction] = None
    argrepr: Optional[str] = None

    # def __hash__(self):
    #     return id(self)

    # def __eq__(self, other):
    #     return id(self) == id(other)

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

    def rewrite(
        self,
        opcode: Optional[int] = None,
        opname: Optional[str] = None,
        arg: Optional[int] = None,
        argval: Optional[Any] = None,
        offset: Optional[int] = None,
        starts_line: Optional[int] = None,
        is_jump_target: bool = False,
        target: Optional[dis.Instruction] = None,
    ):
        return Instruction(
            self.opcode if opcode is None else opcode,
            self.opname if opname is None else opname,
            self.arg if arg is None else arg,
            self.argval if argval is None else argval,
            self.offset if offset is None else offset,
            self.starts_line if starts_line is None else starts_line,
            self.is_jump_target if is_jump_target is None else is_jump_target,
            self.target if target is None else target,
        )
        pass


def create_instruction(name, arg=None, argval=None, target=None):
    return Instruction.create(name, arg, argval, target)
