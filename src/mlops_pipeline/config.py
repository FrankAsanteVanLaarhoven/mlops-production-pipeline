"""Typed, YAML-backed configuration for every pipeline stage.

All thresholds, search spaces, and paths live here instead of being hardcoded in
pipeline code, so a run is fully described by one config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    source: Literal["synthetic", "csv"] = "synthetic"
    csv_path: Path | None = None
    target_column: str = "target"
    n_samples: int = Field(2000, gt=0)
    n_features: int = Field(10, gt=0)
    test_size: float = Field(0.2, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _csv_requires_path(self) -> DataConfig:
        if self.source == "csv" and self.csv_path is None:
            raise ValueError("data.csv_path is required when data.source is 'csv'")
        return self


class FloatRange(BaseModel):
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> FloatRange:
        if self.low > self.high:
            raise ValueError(f"range low ({self.low}) must be <= high ({self.high})")
        return self


class IntRange(BaseModel):
    low: int
    high: int

    @model_validator(mode="after")
    def _ordered(self) -> IntRange:
        if self.low > self.high:
            raise ValueError(f"range low ({self.low}) must be <= high ({self.high})")
        return self


class TrainingConfig(BaseModel):
    n_trials: int = Field(20, gt=0, description="Optuna trials for hyperparameter search")
    hpo_epochs: int = Field(50, gt=0, description="epochs per Optuna trial")
    epochs: int = Field(100, gt=0, description="epochs for the final model")
    lr: FloatRange = FloatRange(low=1e-3, high=1e-1, log=True)
    weight_bit_width: IntRange = IntRange(low=4, high=8)
    hidden_dim: IntRange = IntRange(low=4, high=32)


class GatesConfig(BaseModel):
    """Quality gates. A run that breaches any gate fails instead of shipping a model."""

    max_drifted_share: float = Field(0.3, ge=0.0, le=1.0)
    min_accuracy: float = Field(0.85, ge=0.0, le=1.0)
    min_noise_consistency: float = Field(0.90, ge=0.0, le=1.0)
    noise_std: float = Field(0.05, gt=0.0)


class RegistryConfig(BaseModel):
    root: Path = Path("artifacts/registry")
    dvc_track: bool = True


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, gt=0, lt=65536)
    max_abs_feature_value: float = Field(
        10.0, gt=0.0, description="reject inputs with |value| above this as out-of-distribution"
    )


class PipelineConfig(BaseModel):
    seed: int = 42
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    gates: GatesConfig = GatesConfig()
    registry: RegistryConfig = RegistryConfig()
    serving: ServingConfig = ServingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def as_step_param(self) -> dict:
        """JSON-safe dict for passing across ZenML step boundaries."""
        return self.model_dump(mode="json")
