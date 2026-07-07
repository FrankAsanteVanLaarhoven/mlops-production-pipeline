import torch

from mlops_pipeline.model import build_model, load_checkpoint, save_checkpoint


def test_outputs_are_probabilities():
    model = build_model(n_features=10, hidden_dim=5, weight_bit_width=6)
    with torch.no_grad():
        out = model(torch.randn(32, 10))
    assert out.shape == (32, 1)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())


def test_checkpoint_roundtrip(tmp_path):
    architecture = {"n_features": 10, "hidden_dim": 7, "weight_bit_width": 5}
    model = build_model(**architecture)
    path = tmp_path / "model.pt"
    save_checkpoint(model, architecture, path)

    restored, restored_arch = load_checkpoint(path)
    assert restored_arch == architecture

    x = torch.randn(16, 10)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
