"""Quantization-aware model factory and checkpoint I/O.

The checkpoint stores the architecture next to the weights so any consumer
(evaluation, registry, serving) can rebuild the exact network without guessing.
"""

from __future__ import annotations

from pathlib import Path

import brevitas.nn as qnn
import torch
import torch.nn as nn


def build_model(n_features: int, hidden_dim: int, weight_bit_width: int) -> nn.Sequential:
    """Binary classifier MLP with quantized (Brevitas) linear layers."""
    return nn.Sequential(
        qnn.QuantLinear(n_features, hidden_dim, weight_bit_width=weight_bit_width, bias=True),
        nn.ReLU(),
        qnn.QuantLinear(hidden_dim, 1, weight_bit_width=weight_bit_width, bias=True),
        nn.Sigmoid(),
    )


def save_checkpoint(model: nn.Module, architecture: dict, path: str | Path) -> None:
    """Persist weights together with the architecture needed to rebuild them."""
    torch.save({"state_dict": model.state_dict(), "architecture": architecture}, path)


def load_checkpoint(path: str | Path) -> tuple[nn.Sequential, dict]:
    """Rebuild the exact network from a checkpoint; returns (model, architecture)."""
    checkpoint = torch.load(path, weights_only=False)
    architecture = checkpoint["architecture"]
    model = build_model(
        n_features=architecture["n_features"],
        hidden_dim=architecture["hidden_dim"],
        weight_bit_width=architecture["weight_bit_width"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, architecture
