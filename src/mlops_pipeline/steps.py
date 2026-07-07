"""ZenML step wrappers.

Each step is a thin adapter: parse config, call the framework-free core function,
return JSON/array data that ZenML can materialize. All real logic lives in
data.py / training.py / validation.py / registry.py where it is unit-tested.
"""

import os
from pathlib import Path

# Must be set before zenml is imported so this repo keeps its own ZenML state.
os.environ.setdefault("ZENML_CONFIG_PATH", str(Path(".zenml-config").resolve()))

from zenml import step

from .config import PipelineConfig
from .data import DatasetBundle, load_dataset
from .model import load_checkpoint, save_checkpoint
from .registry import register_model
from .training import run_hpo, train_final
from .validation import (
    data_drift_share,
    enforce_data_gate,
    enforce_model_gates,
    evaluate_model,
)

DRIFT_REPORT_PATH = Path("artifacts/reports/data_drift_report.html")
STAGING_CHECKPOINT = Path("artifacts/staging/model.pt")


@step
def ingest_step(config: dict) -> dict:
    cfg = PipelineConfig.model_validate(config)
    bundle = load_dataset(cfg.data, cfg.seed)
    print(
        f"[ingest] source={cfg.data.source} "
        f"train={bundle.X_train.shape} test={bundle.X_test.shape} "
        f"fingerprint={bundle.fingerprint()}"
    )
    return bundle.to_dict()


@step
def data_gate_step(config: dict, dataset: dict) -> float:
    cfg = PipelineConfig.model_validate(config)
    bundle = DatasetBundle.from_dict(dataset)
    share = data_drift_share(bundle, report_path=DRIFT_REPORT_PATH)
    print(f"[data-gate] drifted feature share: {share:.2%} (report: {DRIFT_REPORT_PATH})")
    enforce_data_gate(share, cfg.gates)
    return share


@step
def train_step(config: dict, dataset: dict, drift_share: float) -> dict:
    cfg = PipelineConfig.model_validate(config)
    bundle = DatasetBundle.from_dict(dataset)

    print(f"[train] Optuna search: {cfg.training.n_trials} trials")
    params = run_hpo(bundle, cfg.training, cfg.seed)
    print(f"[train] best hyperparameters: {params}")

    model = train_final(bundle, params, cfg.training, cfg.seed)
    architecture = {
        "n_features": bundle.n_features,
        "hidden_dim": params["hidden_dim"],
        "weight_bit_width": params["weight_bit_width"],
    }
    STAGING_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, architecture, STAGING_CHECKPOINT)
    print(f"[train] staged checkpoint at {STAGING_CHECKPOINT}")
    return {
        "checkpoint_path": str(STAGING_CHECKPOINT),
        "params": params,
        "architecture": architecture,
    }


@step
def evaluate_step(config: dict, dataset: dict, training_output: dict) -> dict:
    cfg = PipelineConfig.model_validate(config)
    bundle = DatasetBundle.from_dict(dataset)
    model, _ = load_checkpoint(training_output["checkpoint_path"])

    metrics = evaluate_model(model, bundle, cfg.gates.noise_std, cfg.seed)
    print(
        f"[evaluate] accuracy={metrics['accuracy']:.2%} "
        f"noise_consistency={metrics['noise_consistency']:.2%} "
        f"outputs_bounded={metrics['outputs_bounded']}"
    )
    enforce_model_gates(metrics, cfg.gates)
    print("[evaluate] all model gates passed")
    return metrics


@step
def register_step(
    config: dict, dataset: dict, training_output: dict, metrics: dict, drift_share: float
) -> str:
    cfg = PipelineConfig.model_validate(config)
    bundle = DatasetBundle.from_dict(dataset)
    model, architecture = load_checkpoint(training_output["checkpoint_path"])

    card = register_model(
        model=model,
        architecture=architecture,
        params=training_output["params"],
        metrics=metrics,
        drift_share=drift_share,
        data_fingerprint=bundle.fingerprint(),
        root=cfg.registry.root,
        dvc_track=cfg.registry.dvc_track,
    )
    print(f"[register] published {card['version']} to {cfg.registry.root}")
    return card["version"]
