"""Hyperparameter search (Optuna) and final training for the quantized classifier."""

from __future__ import annotations

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from .config import TrainingConfig
from .data import DatasetBundle
from .model import build_model


def _fit(model: nn.Module, X: torch.Tensor, y: torch.Tensor, lr: float, epochs: int) -> float:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return float(loss.item())


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
    return float((preds == y).float().mean().item())


def run_hpo(bundle: DatasetBundle, cfg: TrainingConfig, seed: int) -> dict:
    """Search lr, bit width, and hidden size; return the best trial's params."""
    X_train = torch.tensor(bundle.X_train)
    y_train = torch.tensor(bundle.y_train)
    X_test = torch.tensor(bundle.X_test)
    y_test = torch.tensor(bundle.y_test)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", cfg.lr.low, cfg.lr.high, log=cfg.lr.log)
        bit_width = trial.suggest_int(
            "weight_bit_width", cfg.weight_bit_width.low, cfg.weight_bit_width.high
        )
        hidden_dim = trial.suggest_int("hidden_dim", cfg.hidden_dim.low, cfg.hidden_dim.high)

        torch.manual_seed(seed)
        model = build_model(bundle.n_features, hidden_dim, bit_width)
        _fit(model, X_train, y_train, lr, cfg.hpo_epochs)
        return accuracy(model, X_test, y_test)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=cfg.n_trials)
    return dict(study.best_trial.params)


def train_final(
    bundle: DatasetBundle, params: dict, cfg: TrainingConfig, seed: int
) -> nn.Sequential:
    """Retrain with the winning hyperparameters for the full epoch budget."""
    torch.manual_seed(seed)
    model = build_model(bundle.n_features, params["hidden_dim"], params["weight_bit_width"])
    _fit(
        model,
        torch.tensor(bundle.X_train),
        torch.tensor(bundle.y_train),
        params["lr"],
        cfg.epochs,
    )
    model.eval()
    return model
