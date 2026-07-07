"""End-to-end training pipeline: ingest → data gate → HPO/train → model gates → register."""

import argparse
import os
from pathlib import Path

# Must be set before zenml is imported so this repo keeps its own ZenML state.
os.environ.setdefault("ZENML_CONFIG_PATH", str(Path(".zenml-config").resolve()))

from zenml import pipeline

from .config import PipelineConfig
from .steps import data_gate_step, evaluate_step, ingest_step, register_step, train_step


@pipeline
def training_pipeline(config: dict):
    dataset = ingest_step(config)
    drift_share = data_gate_step(config, dataset)
    trained = train_step(config, dataset, drift_share)
    metrics = evaluate_step(config, dataset, trained)
    register_step(config, dataset, trained, metrics, drift_share)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MLOps training pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="path to the pipeline YAML config (default: configs/pipeline.yaml)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    print(f"=== training pipeline starting (config: {args.config}, seed: {cfg.seed}) ===")
    training_pipeline(config=cfg.as_step_param())
    print(f"=== training pipeline complete; registry: {cfg.registry.root} ===")


if __name__ == "__main__":
    main()
