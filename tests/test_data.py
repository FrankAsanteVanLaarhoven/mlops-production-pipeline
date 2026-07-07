import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.config import DataConfig
from mlops_pipeline.data import DatasetBundle, load_dataset


def test_synthetic_shapes(config, bundle):
    n = config.data.n_samples
    n_test = int(round(n * config.data.test_size))
    assert bundle.X_train.shape == (n - n_test, config.data.n_features)
    assert bundle.X_test.shape == (n_test, config.data.n_features)
    assert bundle.y_train.shape == (n - n_test, 1)
    assert set(np.unique(bundle.y_train)) <= {0.0, 1.0}


def test_same_seed_is_deterministic(config):
    a = load_dataset(config.data, seed=123)
    b = load_dataset(config.data, seed=123)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    assert a.fingerprint() == b.fingerprint()


def test_different_seed_differs(config):
    a = load_dataset(config.data, seed=1)
    b = load_dataset(config.data, seed=2)
    assert a.fingerprint() != b.fingerprint()


def test_dict_roundtrip(bundle):
    restored = DatasetBundle.from_dict(bundle.to_dict())
    assert restored.fingerprint() == bundle.fingerprint()
    assert restored.feature_names == bundle.feature_names


def test_csv_loading(tmp_path):
    df = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3, 0.4, 0.5], "b": [1.0, 0.9, 0.8, 0.7, 0.6], "target": [0, 1, 0, 1, 0]}
    )
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    cfg = DataConfig(source="csv", csv_path=csv, test_size=0.2)
    bundle = load_dataset(cfg, seed=0)
    assert bundle.feature_names == ["a", "b"]
    assert bundle.X_train.shape == (4, 2)
    assert bundle.X_test.shape == (1, 2)


def test_csv_missing_target_column(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    cfg = DataConfig(source="csv", csv_path=csv)
    with pytest.raises(ValueError, match="target column"):
        load_dataset(cfg, seed=0)
