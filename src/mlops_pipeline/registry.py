"""Versioned model registry with lineage.

Each registered model gets an immutable version directory containing the checkpoint
and a model card (metrics, hyperparameters, data fingerprint, git commit, framework
versions). A `latest.json` pointer selects the serving model; DVC optionally tracks
the whole registry for remote storage.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch.nn as nn

from .model import load_checkpoint, save_checkpoint

MODEL_FILENAME = "model.pt"
CARD_FILENAME = "card.json"
LATEST_FILENAME = "latest.json"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _framework_versions() -> dict:
    import brevitas
    import torch

    return {"torch": torch.__version__, "brevitas": brevitas.__version__}


def register_model(
    model: nn.Module,
    architecture: dict,
    params: dict,
    metrics: dict,
    drift_share: float,
    data_fingerprint: str,
    root: str | Path,
    dvc_track: bool = False,
    feature_names: list[str] | None = None,
) -> dict:
    """Persist the model with its card, update the latest pointer, return the card."""
    root = Path(root)
    commit = _git_commit()
    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    if commit:
        version = f"{version}-{commit}"

    version_dir = root / version
    version_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(model, architecture, version_dir / MODEL_FILENAME)

    card = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "data_fingerprint": data_fingerprint,
        "feature_names": feature_names,
        "architecture": architecture,
        "hyperparameters": params,
        "metrics": metrics,
        "drift_share": drift_share,
        "framework_versions": _framework_versions(),
    }
    (version_dir / CARD_FILENAME).write_text(json.dumps(card, indent=2))
    (root / LATEST_FILENAME).write_text(json.dumps({"version": version}, indent=2))

    if dvc_track:
        _dvc_track(root)
    return card


def _dvc_track(root: Path) -> None:
    if shutil.which("dvc") is None:
        print("[registry] dvc executable not found; skipping artifact versioning")
        return
    result = subprocess.run(["dvc", "add", str(root)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dvc add failed: {result.stderr.strip()}")
    print(f"[registry] DVC tracking updated for {root}")


def load_latest(root: str | Path) -> tuple[nn.Module, dict]:
    """Load the model and card pointed at by latest.json."""
    root = Path(root)
    pointer = root / LATEST_FILENAME
    if not pointer.exists():
        raise FileNotFoundError(f"no latest pointer at {pointer}; run the training pipeline first")
    version = json.loads(pointer.read_text())["version"]
    version_dir = root / version
    model, _ = load_checkpoint(version_dir / MODEL_FILENAME)
    card = json.loads((version_dir / CARD_FILENAME).read_text())
    return model, card
