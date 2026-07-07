import pytest
from pydantic import ValidationError

from mlops_pipeline.schemas import (
    PredictionRequest,
    PredictionResponse,
    validate_feature_vector,
)


def test_request_accepts_feature_list():
    req = PredictionRequest(features=[0.1, 0.2])
    assert req.features == [0.1, 0.2]


def test_request_rejects_non_numeric():
    with pytest.raises(ValidationError):
        PredictionRequest(features=["a", "b"])


def test_feature_vector_length_check():
    with pytest.raises(ValueError, match="expected 10 features"):
        validate_feature_vector([0.1, 0.2], expected_length=10, max_abs_value=10.0)


def test_feature_vector_ood_check():
    with pytest.raises(ValueError, match="out-of-distribution"):
        validate_feature_vector([15.5] + [0.0] * 9, expected_length=10, max_abs_value=10.0)


def test_feature_vector_valid_passes():
    validate_feature_vector([0.5] * 10, expected_length=10, max_abs_value=10.0)


def test_response_bounds():
    PredictionResponse(predicted_class=1, probability=0.9, model_version="v1")
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_class=2, probability=0.9, model_version="v1")
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_class=0, probability=1.5, model_version="v1")
