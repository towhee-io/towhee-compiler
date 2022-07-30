from ..passes import TERMINAL_OPCODES
from ..passes import JUMP_OPCODES


def remove_dead_code(instructions):
    """Dead code elimination"""
    indexof = {id(inst): i for i, inst in enumerate(instructions)}
    live_code = set()

    def find_live_code(start):
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.opcode in JUMP_OPCODES:
                find_live_code(indexof[id(inst.target)])
            if inst.opcode in TERMINAL_OPCODES:
                return

    find_live_code(0)
    return [inst for i, inst in enumerate(instructions) if i in live_code]
