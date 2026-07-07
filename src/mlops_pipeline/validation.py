"""Quality gates: data drift, model performance, robustness fuzzing, boundary safety.

Every gate raises GateFailure with the full list of breaches, so a failed run
reports everything wrong at once instead of one breach per rerun.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import GatesConfig
from .data import DatasetBundle

DRIFTED_COLUMNS_METRIC = "evidently:metric_v2:DriftedColumnsCount"


class GateFailure(RuntimeError):
    """A quality gate was breached; the pipeline must not ship this model."""


def data_drift_share(bundle: DatasetBundle, report_path: str | Path | None = None) -> float:
    """Run Evidently drift + summary presets; return the share of drifted features."""
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    reference, current = bundle.as_frames()
    report = Report(metrics=[DataSummaryPreset(), DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        result.save_html(str(report_path))

    metrics = result.dict()["metrics"]
    drift_metric = next(
        m for m in metrics if m.get("config", {}).get("type") == DRIFTED_COLUMNS_METRIC
    )
    return float(drift_metric["value"]["share"])


def evaluate_model(
    model: nn.Module, bundle: DatasetBundle, noise_std: float, seed: int
) -> dict:
    """Accuracy, perturbation consistency, and output-bound safety in one pass."""
    X = torch.tensor(bundle.X_test)
    y = torch.tensor(bundle.y_test)

    model.eval()
    with torch.no_grad():
        probs = model(X)
        preds = (probs > 0.5).float()
        acc = float((preds == y).float().mean().item())

        torch.manual_seed(seed)
        noise = torch.randn_like(X) * noise_std
        fuzzed_preds = (model(X + noise) > 0.5).float()
        consistency = float((preds == fuzzed_preds).float().mean().item())

        extremes = torch.tensor(
            [[1e6] * bundle.n_features, [-1e6] * bundle.n_features, [0.0] * bundle.n_features],
            dtype=torch.float32,
        )
        extreme_outputs = model(extremes)
        bounded = bool(((extreme_outputs >= 0.0) & (extreme_outputs <= 1.0)).all().item())

    return {
        "accuracy": acc,
        "noise_consistency": consistency,
        "outputs_bounded": bounded,
        "noise_std": noise_std,
    }


def enforce_data_gate(drift_share: float, gates: GatesConfig) -> None:
    """Raise GateFailure when the drifted-feature share breaches the limit."""
    if drift_share >= gates.max_drifted_share:
        raise GateFailure(
            f"data gate breached: drifted feature share {drift_share:.2%} "
            f">= limit {gates.max_drifted_share:.2%}"
        )


def enforce_model_gates(metrics: dict, gates: GatesConfig) -> None:
    """Raise GateFailure listing every breached model gate at once."""
    breaches = []
    if metrics["accuracy"] < gates.min_accuracy:
        breaches.append(
            f"accuracy {metrics['accuracy']:.2%} < required {gates.min_accuracy:.2%}"
        )
    if metrics["noise_consistency"] < gates.min_noise_consistency:
        breaches.append(
            f"noise consistency {metrics['noise_consistency']:.2%} "
            f"< required {gates.min_noise_consistency:.2%}"
        )
    if not metrics["outputs_bounded"]:
        breaches.append("output probabilities escape [0, 1] on extreme inputs")
    if breaches:
        raise GateFailure("model gates breached: " + "; ".join(breaches))
