"""End-to-end training pipeline entry point."""

import argparse

from .adapters.zenml_pipeline import ZenMLPipelineAdapter
from .core.pipeline import BasePipeline

# Registry of available orchestrators
_ORCHESTRATORS: dict[str, type[BasePipeline]] = {
    "zenml": ZenMLPipelineAdapter,
}


def register_orchestrator(name: str, orchestrator_cls: type[BasePipeline]) -> None:
    """Register a custom orchestration engine (e.g. airflow, prefect)."""
    _ORCHESTRATORS[name] = orchestrator_cls


def get_pipeline_orchestrator(name: str = "zenml") -> BasePipeline:
    """Retrieve an orchestration engine instance by name."""
    if name not in _ORCHESTRATORS:
        raise ValueError(
            f"Orchestrator '{name}' is not registered. Choose from: {list(_ORCHESTRATORS.keys())}"
        )
    return _ORCHESTRATORS[name]()


def main() -> None:
    """CLI entry point for `mlops-train`."""
    parser = argparse.ArgumentParser(description="Run the MLOps training pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="path to the pipeline YAML config (default: configs/pipeline.yaml)",
    )
    parser.add_argument(
        "--engine",
        default="zenml",
        help="orchestrator engine to use (default: zenml)",
    )
    args = parser.parse_args()

    orchestrator = get_pipeline_orchestrator(args.engine)
    orchestrator.run(args.config)


if __name__ == "__main__":
    main()
