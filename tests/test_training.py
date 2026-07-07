import torch

from mlops_pipeline.training import accuracy, run_hpo


def test_hpo_returns_params_within_search_space(config, bundle):
    params = run_hpo(bundle, config.training, config.seed)
    t = config.training
    assert t.lr.low <= params["lr"] <= t.lr.high
    assert t.weight_bit_width.low <= params["weight_bit_width"] <= t.weight_bit_width.high
    assert t.hidden_dim.low <= params["hidden_dim"] <= t.hidden_dim.high


def test_final_model_learns_the_task(trained, bundle):
    model, _ = trained
    acc = accuracy(
        model, torch.tensor(bundle.X_test), torch.tensor(bundle.y_test)
    )
    # Linearly separable synthetic task: anything near chance means training is broken.
    assert acc >= 0.8, f"final model accuracy {acc:.2%} is too low"
