import dis
from dis import Instruction
from typing import List


def remove_load_call_method(instructions: List[Instruction]):
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    rewrites = {"LOAD_METHOD": "LOAD_ATTR", "CALL_METHOD": "CALL_FUNCTION"}
    for inst in instructions:
        if inst.opname in rewrites:
            inst.opname=rewrites[inst.opname]
            inst.opcode=dis.opmap[inst.opname]
    return instructions
