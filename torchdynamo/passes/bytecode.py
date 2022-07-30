from .pass_manager import PassManager
from ..passes.remove_dead_code import remove_dead_code
from ..passes.remove_pointless_jumps import remove_pointless_jumps
from .. import config


def common():
    if config.dead_code_elimination:
        return PassManager().add(remove_dead_code).add(remove_pointless_jumps)
    return PassManager()
