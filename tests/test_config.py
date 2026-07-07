import pytest
from pydantic import ValidationError

from mlops_pipeline.config import PipelineConfig


def test_defaults_are_valid():
    cfg = PipelineConfig()
    assert cfg.data.source == "synthetic"
    assert cfg.gates.min_accuracy == 0.85
    assert cfg.registry.dvc_track is True


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        """
seed: 11
data: { n_samples: 100, n_features: 4 }
training: { n_trials: 3 }
gates: { min_accuracy: 0.5 }
"""
    )
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.seed == 11
    assert cfg.data.n_features == 4
    assert cfg.training.n_trials == 3
    assert cfg.gates.min_accuracy == 0.5


def test_csv_source_requires_path():
    with pytest.raises(ValidationError, match="csv_path"):
        PipelineConfig.model_validate({"data": {"source": "csv"}})


def test_invalid_range_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {"training": {"weight_bit_width": {"low": 8, "high": 4}}}
        )


def test_step_param_is_json_safe():
    import json

    json.dumps(PipelineConfig().as_step_param())
