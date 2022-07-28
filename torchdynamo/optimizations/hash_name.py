import base64
import hashlib
import io
import itertools
import logging
import os
from collections import defaultdict

import torch

from .. import config
from .normalize import long_name

log = logging.getLogger(__name__)


def string_key(gm: torch.fx.GraphModule, example_inputs):
    out = io.StringIO()
    node_to_id = defaultdict(iter(itertools.count()).__next__)

    def argkey(n: torch.fx.Node):
        return f"#{node_to_id[n]}"

    def tensorkey(t):
        if isinstance(t, torch.Tensor):
            requires_grad = t.requires_grad and torch.torch.is_grad_enabled()
            return (
                f"{t.__class__.__name__}({t.dtype}, {t.device}, "
                f"{tuple(t.size())}, {tuple(t.stride())}, {requires_grad})"
            )
        return type(t).__name__

    inputs_iter = iter(example_inputs)

    for node in gm.graph.nodes:
        key = argkey(node)
        name = "."
        if node.op == "placeholder":
            name = tensorkey(next(inputs_iter))
        elif node.op == "get_attr":
            val = eval(f"self.{node.target}", {"self": gm})
            name = tensorkey(val)
        elif node.op in ("call_function", "call_method", "call_module"):
            name = long_name(gm, node)
        out.write(
            f"{key} {node.op} {name} "
            f"{torch.fx.map_arg(node.args, argkey)!r} "
            f"{torch.fx.map_arg(node.kwargs, argkey)!r}\n"
        )
    return out.getvalue()


def graph_hash(gm: torch.fx.GraphModule, example_inputs):
    return "g" + base64.urlsafe_b64encode(
        hashlib.sha256(string_key(gm, example_inputs).encode("utf-8")).digest()
    )[:39].decode("utf-8")


def folder_name(gm: torch.fx.GraphModule, example_inputs):
    base = os.path.join(config.base_dir, "subgraphs")
    if not os.path.exists(base):
        os.mkdir(base)
        open(os.path.join(base, "__init__.py"), "w").close()
    return os.path.join(base, graph_hash(gm, example_inputs))
