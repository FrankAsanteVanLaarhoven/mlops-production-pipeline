import json

import torch

from mlops_pipeline.registry import load_latest, register_model


def _register(trained, bundle, root):
    model, params = trained
    architecture = {
        "n_features": bundle.n_features,
        "hidden_dim": params["hidden_dim"],
        "weight_bit_width": params["weight_bit_width"],
    }
    return register_model(
        model=model,
        architecture=architecture,
        params=params,
        metrics={"accuracy": 0.95, "noise_consistency": 0.97, "outputs_bounded": True},
        drift_share=0.0,
        data_fingerprint=bundle.fingerprint(),
        root=root,
        dvc_track=False,
    )


def test_register_writes_card_and_pointer(trained, bundle, tmp_path):
    root = tmp_path / "registry"
    card = _register(trained, bundle, root)

    version_dir = root / card["version"]
    assert (version_dir / "model.pt").exists()

    saved_card = json.loads((version_dir / "card.json").read_text())
    assert saved_card["data_fingerprint"] == bundle.fingerprint()
    assert saved_card["metrics"]["accuracy"] == 0.95
    assert saved_card["hyperparameters"] == card["hyperparameters"]
    assert saved_card["framework_versions"]["torch"]

    pointer = json.loads((root / "latest.json").read_text())
    assert pointer["version"] == card["version"]


def test_load_latest_roundtrip(trained, bundle, tmp_path):
    root = tmp_path / "registry"
    card = _register(trained, bundle, root)

    model, loaded_card = load_latest(root)
    assert loaded_card["version"] == card["version"]

    x = torch.tensor(bundle.X_test[:8])
    with torch.no_grad():
        torch.testing.assert_close(trained[0](x), model(x))


def test_load_latest_missing_pointer(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="latest pointer"):
        load_latest(tmp_path / "empty")


def test_dvc_track_skips_when_dvc_missing(trained, bundle, tmp_path, monkeypatch, capsys):
    import mlops_pipeline.registry as registry_module

    monkeypatch.setattr(registry_module.shutil, "which", lambda _: None)
    model, params = trained
    architecture = {
        "n_features": bundle.n_features,
        "hidden_dim": params["hidden_dim"],
        "weight_bit_width": params["weight_bit_width"],
    }
    register_model(
        model=model,
        architecture=architecture,
        params=params,
        metrics={"accuracy": 0.9},
        drift_share=0.0,
        data_fingerprint=bundle.fingerprint(),
        root=tmp_path / "registry",
        dvc_track=True,
    )
    assert "skipping artifact versioning" in capsys.readouterr().out


def test_dvc_track_raises_on_dvc_failure(trained, bundle, tmp_path, monkeypatch):
    import subprocess

    import pytest

    import mlops_pipeline.registry as registry_module

    monkeypatch.setattr(registry_module.shutil, "which", lambda _: "/usr/bin/dvc")
    monkeypatch.setattr(
        registry_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=1, stdout="", stderr="boom"),
    )
    model, params = trained
    architecture = {
        "n_features": bundle.n_features,
        "hidden_dim": params["hidden_dim"],
        "weight_bit_width": params["weight_bit_width"],
    }
    with pytest.raises(RuntimeError, match="dvc add failed"):
        register_model(
            model=model,
            architecture=architecture,
            params=params,
            metrics={"accuracy": 0.9},
            drift_share=0.0,
            data_fingerprint=bundle.fingerprint(),
            root=tmp_path / "registry",
            dvc_track=True,
        )


def test_git_commit_failure_returns_none(trained, bundle, tmp_path, monkeypatch):
    import subprocess
    import mlops_pipeline.registry as registry_module

    def mock_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(registry_module.subprocess, "run", mock_run)

    model, params = trained
    architecture = {
        "n_features": bundle.n_features,
        "hidden_dim": params["hidden_dim"],
        "weight_bit_width": params["weight_bit_width"],
    }
    card = register_model(
        model=model,
        architecture=architecture,
        params=params,
        metrics={"accuracy": 0.9},
        drift_share=0.0,
        data_fingerprint=bundle.fingerprint(),
        root=tmp_path / "registry",
        dvc_track=False,
    )
    assert card["git_commit"] is None


def test_dvc_track_success_prints_message(trained, bundle, tmp_path, monkeypatch, capsys):
    import subprocess
    import mlops_pipeline.registry as registry_module

    monkeypatch.setattr(registry_module.shutil, "which", lambda _: "/usr/bin/dvc")
    monkeypatch.setattr(
        registry_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="ok", stderr=""),
    )

    model, params = trained
    architecture = {
        "n_features": bundle.n_features,
        "hidden_dim": params["hidden_dim"],
        "weight_bit_width": params["weight_bit_width"],
    }
    register_model(
        model=model,
        architecture=architecture,
        params=params,
        metrics={"accuracy": 0.9},
        drift_share=0.0,
        data_fingerprint=bundle.fingerprint(),
        root=tmp_path / "registry",
        dvc_track=True,
    )
    assert "DVC tracking updated for" in capsys.readouterr().out

