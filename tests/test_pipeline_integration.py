"""Full-stack pipeline run through ZenML. Marked integration: needs the ZenML
client initialised and takes minutes rather than seconds.

Run with: pytest -m integration
"""

import json

import pytest

zenml = pytest.importorskip("zenml")

pytestmark = pytest.mark.integration


def test_training_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from mlops_pipeline.adapters.zenml_pipeline import zenml_training_pipeline
    from mlops_pipeline.config import PipelineConfig

    cfg = PipelineConfig.model_validate(
        {
            "seed": 7,
            "data": {"n_samples": 400, "n_features": 10},
            "training": {"n_trials": 2, "hpo_epochs": 15, "epochs": 60},
            "registry": {"root": str(tmp_path / "registry"), "dvc_track": False},
        }
    )
    zenml_training_pipeline(config=cfg.as_step_param())

    pointer = tmp_path / "registry" / "latest.json"
    assert pointer.exists()
    version = json.loads(pointer.read_text())["version"]
    card = json.loads((tmp_path / "registry" / version / "card.json").read_text())
    assert card["metrics"]["accuracy"] >= 0.85
