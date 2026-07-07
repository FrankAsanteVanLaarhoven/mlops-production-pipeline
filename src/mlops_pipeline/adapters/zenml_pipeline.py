"""ZenML pipeline adapter implementation."""

import os
from pathlib import Path

from ..config import PipelineConfig
from ..core.pipeline import BasePipeline

# Must be set before zenml is imported so this repo keeps its own ZenML state.
os.environ.setdefault("ZENML_CONFIG_PATH", str(Path(".zenml-config").resolve()))

from zenml import pipeline

from ..steps import data_gate_step, evaluate_step, ingest_step, register_step, train_step


@pipeline
def zenml_training_pipeline(config: dict) -> None:
    """Gated lifecycle: no model is registered unless every quality gate passes."""
    dataset = ingest_step(config)
    drift_share = data_gate_step(config, dataset)
    trained = train_step(config, dataset, drift_share)
    metrics = evaluate_step(config, dataset, trained)
    register_step(config, dataset, trained, metrics, drift_share)


class ZenMLPipelineAdapter(BasePipeline):
    """ZenML orchestrator implementation of BasePipeline."""

    def run(self, config_path: str) -> None:
        """Run the full gated training/selection lifecycle via ZenML."""
        cfg = PipelineConfig.from_yaml(config_path)
        print(f"=== training pipeline starting (config: {config_path}, seed: {cfg.seed}) ===")
        zenml_training_pipeline(config=cfg.as_step_param())
        print(f"=== training pipeline complete; registry: {cfg.registry.root} ===")
