from .pass_manager import PassManager
from ..passes import remove_dead_code
from ..passes import remove_pointless_jump
from .. import config


def common():
    if config.dead_code_elimination:
        return PassManager().add(remove_dead_code).add(remove_pointless_jump)
    return PassManager()
