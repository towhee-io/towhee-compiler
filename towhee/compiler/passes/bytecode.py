from torchdynamo import config

from .pass_manager import PassManager
from .remove_dead_code import remove_dead_code
from .remove_load_call_method import remove_load_call_method
from .remove_pointless_jumps import remove_pointless_jumps
from .virtualize_jumps import virtualize_jumps

__all__ = [
    "common",
    "remove_dead_code",
    "remove_load_call_method",
    "remove_pointless_jumps",
    "virtualize_jumps",
]


def common():
    if config.dead_code_elimination:
        return PassManager().add(remove_dead_code).add(remove_pointless_jumps)
    return PassManager()
