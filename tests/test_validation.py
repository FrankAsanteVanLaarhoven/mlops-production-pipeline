import pytest

from mlops_pipeline.config import GatesConfig
from mlops_pipeline.validation import (
    GateFailure,
    data_drift_metrics,
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


def test_data_drift_metrics_structure(bundle):
    share, p_values = data_drift_metrics(bundle)
    assert 0.0 <= share <= 1.0
    assert isinstance(p_values, dict)
    assert len(p_values) > 0
    for col, val in p_values.items():
        assert isinstance(col, str)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


def test_enforce_data_gate_per_column():
    # Pass check: p-value above threshold
    gates = GatesConfig(per_column_drift={"feature_0": 0.05})
    enforce_data_gate(0.1, gates, {"feature_0": 0.2})

    # Fail check: p-value below threshold
    with pytest.raises(GateFailure, match="column 'feature_0' drift p-value"):
        enforce_data_gate(0.1, gates, {"feature_0": 0.01})

    # Fail check: both global share and per-column breach reported
    gates_both = GatesConfig(max_drifted_share=0.2, per_column_drift={"feature_0": 0.05})
    with pytest.raises(GateFailure) as excinfo:
        enforce_data_gate(0.5, gates_both, {"feature_0": 0.01})
    message = str(excinfo.value)
    assert "drifted feature share" in message
    assert "column 'feature_0'" in message
