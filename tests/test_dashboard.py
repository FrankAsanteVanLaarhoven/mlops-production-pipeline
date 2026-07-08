import os
import sys

# Add project root to path so we can import dashboard.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from dashboard import app


@pytest.fixture(autouse=True)
def mock_ray_serve_offline(monkeypatch):
    import httpx
    import dashboard

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, *args, **kwargs):
            raise httpx.ConnectError("offline")

        async def post(self, url, *args, **kwargs):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(dashboard.httpx, "AsyncClient", MockAsyncClient)


@pytest.fixture
def client():
    return TestClient(app)


def test_api_status_local_fallback(client, tmp_path, monkeypatch):
    import dashboard
    
    # Force registry and default model to be non-existent/empty
    # We patch ROOT_DIR in dashboard module
    monkeypatch.setattr(dashboard, "ROOT_DIR", tmp_path)
    
    # Case 1: Neither exists -> offline
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["serve_status"] == "offline"
    assert data["model_version"] == "None"
    
    # Case 2: Create default model folder and files
    default_model_dir = tmp_path / "src" / "mlops_pipeline" / "default_model"
    default_model_dir.mkdir(parents=True)
    (default_model_dir / "latest.json").write_text('{"version": "."}')
    (default_model_dir / "card.json").write_text(
        '{"version": "v_test_default", "metrics": {"accuracy": 0.99}}'
    )
    
    # We mock load_latest since we don't have a real model.pt in tmp_path
    from mlops_pipeline.registry import load_latest
    monkeypatch.setattr(
        "mlops_pipeline.registry.load_latest",
        lambda root: (None, {"version": "v_test_default", "metrics": {"accuracy": 0.99}}),
    )
    
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["serve_status"] == "online"
    assert data["model_version"] == "v_test_default"
    assert data["metrics"]["accuracy"] == 0.99


def test_api_predict_local_fallback(client, tmp_path, monkeypatch):
    import torch
    import torch.nn as nn
    import dashboard
    
    # Mock load_latest to return a dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            # return 0.9 for any input
            return torch.tensor([0.9])
            
    dummy_model = DummyModel()
    dummy_card = {
        "version": "v_test_default",
        "architecture": {"n_features": 10},
    }
    
    monkeypatch.setattr(dashboard, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        "mlops_pipeline.registry.load_latest",
        lambda root: (dummy_model, dummy_card),
    )
    
    # Make sure default_model/latest.json exists to pass the check
    default_model_dir = tmp_path / "src" / "mlops_pipeline" / "default_model"
    default_model_dir.mkdir(parents=True)
    (default_model_dir / "latest.json").write_text('{"version": "."}')
    
    # 1. Valid prediction
    resp = client.post("/api/predict", json={"features": [0.0] * 10})
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_class"] == 1
    assert abs(data["probability"] - 0.9) < 1e-5
    assert data["model_version"] == "v_test_default"
    
    # 2. Invalid features length
    resp = client.post("/api/predict", json={"features": [0.0] * 5})
    assert resp.status_code == 400
    assert "expected 10 features" in resp.json()["details"]
    
    # 3. Out of bounds features
    resp = client.post("/api/predict", json={"features": [15.0] + [0.0] * 9})
    assert resp.status_code == 400
    assert "out-of-distribution" in resp.json()["details"]


def test_api_train_blocked_on_vercel(client, monkeypatch):
    # Set VERCEL environment variable
    monkeypatch.setenv("VERCEL", "1")
    resp = client.post("/api/train")
    assert resp.status_code == 400
    assert "cannot be executed in serverless environments" in resp.json()["message"]


def test_api_train_local_success(client, monkeypatch, tmp_path):
    import subprocess
    import dashboard
    
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(dashboard, "ROOT_DIR", tmp_path)
    
    # Mock subprocess.Popen
    class DummyProcess:
        pid = 1234
        def poll(self):
            return None
            
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: DummyProcess(),
    )
    
    resp = client.post("/api/train")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
    assert resp.json()["pid"] == 1234
