import dis
from dataclasses import dataclass
from typing import Any, Optional

instruction_counter = 0

@dataclass(frozen=True)
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
    target: Optional[dis.Instruction] = None
    _id: int = None
    
    def __post_init__(self):
        global instruction_counter
        self._id = instruction_counter
        instruction_counter += 1

    def __hash__(self):
        return self._id

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
