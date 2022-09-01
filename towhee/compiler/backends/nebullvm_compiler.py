from pathlib import Path

from torchdynamo import config
from torchdynamo.optimizations.subgraph import SubGraph

from .backend_compiler import BackendCompiler


class NebullvmCompiler(BackendCompiler):
    def __init__(self) -> None:
        super().__init__()

    def compile(self, subgraph: SubGraph):
        from nebullvm import optimize_torch_model
        from nebullvm.inference_learners.base import LearnerMetadata

        model = subgraph.model
        inputs = subgraph.example_inputs
        hash_path = subgraph.hash_path
        cached_model_dir = Path(config.cached_dir) / hash_path
        str_cached_model_dir = str(cached_model_dir.absolute())

        from towhee.functional import param_scope

        with param_scope() as ps:
            if cached_model_dir.exists():
                print(f"using cached model in {str_cached_model_dir}")
                return LearnerMetadata.read(str_cached_model_dir + '/optimized_model').load_model(str_cached_model_dir)
            else:
                try:
                    if config.debug:
                        print(f"Saving the model to {str_cached_model_dir}")
                    cached_model_dir.mkdir(parents=True)
                    return optimize_torch_model(
                        model=model,
                        save_dir=str_cached_model_dir,
                        dataloader=[[inputs, None]],
                        perf_loss_ths=ps().towhee.compiler.perf_loss_ths(None),
                    )
                except Exception as e:
                    if config.debug:
                        print(f"Failed to save the model, error:", e)
                    cached_model_dir.rmdir()


BackendCompiler.backends["nebullvm"] = NebullvmCompiler
