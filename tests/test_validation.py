import pytest

from mlops_pipeline.config import GatesConfig
from mlops_pipeline.validation import (
    GateFailure,
    data_drift_share,
    enforce_data_gate,
    enforce_model_gates,
    evaluate_model,
)


def test_drift_share_low_for_iid_split(bundle, tmp_path):
    report = tmp_path / "drift.html"
    share = data_drift_share(bundle, report_path=report)
    assert 0.0 <= share < 0.3
    assert report.exists()


def test_evaluate_model_metrics(trained, bundle, config):
    model, _ = trained
    metrics = evaluate_model(model, bundle, config.gates.noise_std, config.seed)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["noise_consistency"] <= 1.0
    assert metrics["outputs_bounded"] is True


def test_data_gate_blocks_drift():
    gates = GatesConfig(max_drifted_share=0.3)
    enforce_data_gate(0.1, gates)
    with pytest.raises(GateFailure, match="data gate"):
        enforce_data_gate(0.5, gates)


def test_model_gates_report_all_breaches():
    gates = GatesConfig(min_accuracy=0.85, min_noise_consistency=0.90)
    bad = {"accuracy": 0.5, "noise_consistency": 0.5, "outputs_bounded": False}
    with pytest.raises(GateFailure) as excinfo:
        enforce_model_gates(bad, gates)
    message = str(excinfo.value)
    assert "accuracy" in message
    assert "noise consistency" in message
    assert "escape" in message


def test_model_gates_pass_good_metrics():
    gates = GatesConfig()
    enforce_model_gates(
        {"accuracy": 0.95, "noise_consistency": 0.97, "outputs_bounded": True}, gates
    )
