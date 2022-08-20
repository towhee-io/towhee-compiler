from pathlib import Path

from torchdynamo import config
from torchdynamo.optimizations.subgraph import SubGraph

from ..log import get_logger
from .backend_compiler import BackendCompiler

log = get_logger(__name__)


class _NebullvmWrapper:
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args, **kwargs):
        retval = self.fn(*args, **kwargs)
        if isinstance(retval, tuple) and len(retval) == 1:
            return retval[0]
        return retval


class NebullvmCompiler(BackendCompiler):
    def __init__(self) -> None:
        super().__init__()

    def compile(self, subgraph: SubGraph):
        from nebullvm import optimize_torch_model
        from nebullvm.inference_learners.onnx import PytorchONNXInferenceLearner

        model = subgraph.model
        inputs = subgraph.example_inputs
        hash_path = subgraph.hash_path
        cached_model_dir = Path(config.cached_dir) / hash_path
        str_cached_model_dir = str(cached_model_dir.absolute())
        subgraph.model_dir = str_cached_model_dir

        from towhee.functional import param_scope

        with param_scope() as ps:
            if cached_model_dir.exists():
                log.info(f"using cached model in {str_cached_model_dir}")
                retval = PytorchONNXInferenceLearner.load(str_cached_model_dir)
            else:
                try:
                    log.debug(f"Saving the model to {str_cached_model_dir}")
                    cached_model_dir.mkdir(parents=True)
                    subgraph.onnx_filename
                    retval = optimize_torch_model(
                        model=model,
                        save_dir=str_cached_model_dir,
                        dataloader=[[inputs, None]],
                        perf_loss_ths=ps().towhee.compiler.perf_loss_ths(None),
                    )
                except Exception as e:
                    log.debug(f"Failed to save the model, error:", e)
                    cached_model_dir.rmdir()
            return _NebullvmWrapper(retval)


BackendCompiler.backends["nebullvm"] = NebullvmCompiler
