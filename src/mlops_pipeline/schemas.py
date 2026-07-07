"""Request/response contracts for the serving layer.

Kept free of Ray imports so the contracts are unit-testable and reusable
(e.g. by client SDKs or contract tests) without the serving stack installed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Inference request: one feature vector."""

    features: list[float] = Field(..., description="Feature vector for one sample")


class PredictionResponse(BaseModel):
    """Inference response contract; probability is hard-bounded to [0, 1]."""

    model_config = ConfigDict(protected_namespaces=())

    predicted_class: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str


def validate_feature_vector(
    features: list[float], expected_length: int, max_abs_value: float
) -> None:
    """Reject malformed or out-of-distribution inputs before they reach the model."""
    if len(features) != expected_length:
        raise ValueError(
            f"expected {expected_length} features, got {len(features)}"
        )
    for i, value in enumerate(features):
        if abs(value) > max_abs_value:
            raise ValueError(
                f"feature[{i}]={value} is out-of-distribution (|value| > {max_abs_value})"
            )
