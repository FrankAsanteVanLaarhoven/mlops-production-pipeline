"""Data loading with a swappable source (synthetic or CSV) and content fingerprinting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class DatasetBundle:
    """Train/test split with feature names and a content fingerprint."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        """Width of the feature matrix."""
        return self.X_train.shape[1]

    def fingerprint(self) -> str:
        """Content hash recorded in the model card for data lineage."""
        digest = hashlib.sha256()
        for arr in (self.X_train, self.y_train, self.X_test, self.y_test):
            digest.update(np.ascontiguousarray(arr).tobytes())
        return digest.hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serializable form for crossing ZenML step boundaries."""
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatasetBundle:
        """Rebuild a bundle from its `to_dict` form."""
        return cls(
            X_train=d["X_train"],
            y_train=d["y_train"],
            X_test=d["X_test"],
            y_test=d["y_test"],
            feature_names=list(d["feature_names"]),
        )

    def as_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(reference, current) DataFrames for drift analysis."""
        ref = pd.DataFrame(self.X_train, columns=self.feature_names)
        ref["target"] = self.y_train
        cur = pd.DataFrame(self.X_test, columns=self.feature_names)
        cur["target"] = self.y_test
        return ref, cur


def make_synthetic(n_samples: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable binary task: label is 1 iff the feature sum is positive."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


def _split(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    n_test = max(1, int(round(len(X) * test_size)))
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def load_dataset(cfg: DataConfig, seed: int) -> DatasetBundle:
    """Load the configured source (synthetic or CSV) and split it reproducibly."""
    if cfg.source == "synthetic":
        X, y = make_synthetic(cfg.n_samples, cfg.n_features, seed)
        feature_names = [f"feature_{i}" for i in range(cfg.n_features)]
    else:
        df = pd.read_csv(cfg.csv_path)
        if cfg.target_column not in df.columns:
            raise ValueError(f"target column '{cfg.target_column}' not found in {cfg.csv_path}")
        feature_names = [c for c in df.columns if c != cfg.target_column]
        X = df[feature_names].to_numpy(dtype=np.float32)
        y = df[cfg.target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train, y_train, X_test, y_test = _split(X, y, cfg.test_size, seed)
    return DatasetBundle(X_train, y_train, X_test, y_test, feature_names)
