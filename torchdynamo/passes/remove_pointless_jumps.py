def remove_pointless_jumps(instructions):
    """Eliminate jumps to the next instruction"""
    pointless_jumps = {
        id(a)
        for a, b in zip(instructions, instructions[1:])
        if a.opname == "JUMP_ABSOLUTE" and a.target is b
    }
    return [inst for inst in instructions if id(inst) not in pointless_jumps]
