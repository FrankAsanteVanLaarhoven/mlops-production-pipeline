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
    acc = accuracy(model, torch.tensor(bundle.X_test), torch.tensor(bundle.y_test))
    # Linearly separable synthetic task: anything near chance means training is broken.
    assert acc >= 0.8, f"final model accuracy {acc:.2%} is too low"


def test_class_weighted_loss_differs_from_unweighted(bundle):
    from mlops_pipeline.model import build_model
    from mlops_pipeline.training import _fit

    # Create an imbalanced bundle/X/y
    X = torch.tensor(bundle.X_train[:10])
    y = torch.tensor([[0.0]] * 9 + [[1.0]])  # 90% class 0, 10% class 1

    torch.manual_seed(42)
    model1 = build_model(bundle.n_features, 8, 8)
    loss_unweighted = _fit(model1, X, y, lr=0.01, epochs=1, class_weighted=False)

    torch.manual_seed(42)
    model2 = build_model(bundle.n_features, 8, 8)
    loss_weighted = _fit(model2, X, y, lr=0.01, epochs=1, class_weighted=True)

    # They should differ because the weighted loss scales up the loss for the single class 1 sample
    assert loss_weighted != loss_unweighted
