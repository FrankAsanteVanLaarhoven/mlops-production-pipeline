import pytest

from mlops_pipeline.config import PipelineConfig
from mlops_pipeline.data import load_dataset
from mlops_pipeline.training import run_hpo, train_final


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    """Small budgets so the whole suite trains in seconds on CPU."""
    return PipelineConfig.model_validate(
        {
            "seed": 7,
            "data": {"n_samples": 400, "n_features": 10},
            "training": {"n_trials": 2, "hpo_epochs": 15, "epochs": 60},
            "registry": {"dvc_track": False},
        }
    )


@pytest.fixture(scope="session")
def bundle(config):
    return load_dataset(config.data, config.seed)


@pytest.fixture(scope="session")
def trained(config, bundle):
    params = run_hpo(bundle, config.training, config.seed)
    model = train_final(bundle, params, config.training, config.seed)
    return model, params
