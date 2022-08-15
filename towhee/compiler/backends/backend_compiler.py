import tempfile
from typing import Callable
from typing import List
from typing import Optional

from torchdynamo.optimizations.subgraph import SubGraph

from ..log import get_logger

log = get_logger(__name__)


class BackendCompiler:
    backends = {}

    def __init__(self) -> None:
        pass

    def __call__(self, graph, example_inputs: Optional[List] = None) -> Callable:
        if graph is None:
            return None

        if not isinstance(graph, SubGraph):
            with tempfile.TemporaryDirectory() as tmp:
                return self.compile(SubGraph(graph, example_inputs, tmp))
        try:
            return self.compile(graph)
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception(f"{self.__class__.__name__} error")
            return None

    def compile(self, subgraph: SubGraph):
        pass
