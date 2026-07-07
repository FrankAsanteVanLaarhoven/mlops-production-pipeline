import os
import pytest
import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn
import pandas as pd

# Evidently imports for data drift and quality validation
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# 1. Setup helper functions for model loading
def build_model(weight_bit_width):
    return nn.Sequential(
        qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
        nn.ReLU(),
        qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
        nn.Sigmoid()
    )

@pytest.fixture
def trained_model():
    checkpoint_path = "best_model.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Model checkpoint not found. Run training script first.")
    
    checkpoint = torch.load(checkpoint_path)
    weight_bit_width = checkpoint['weight_bit_width']
    
    model = build_model(weight_bit_width)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

@pytest.fixture
def eval_dataset():
    # Generate deterministic evaluation data (set random seed)
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    Y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, Y

# ==========================================
# TEST CATEGORY 1: Data Quality & Schema Checks (Evidently)
# ==========================================

def test_data_quality_report(eval_dataset):
    """Fuzzy Labs MLOps standard: Auto-validate data quality and drift report."""
    X, Y = eval_dataset
    
    # Create pandas dataframes for reference/current data comparison
    feature_names = [f"feature_{i}" for i in range(10)]
    df_reference = pd.DataFrame(X[:100], columns=feature_names)
    df_reference['target'] = Y[:100]
    
    df_current = pd.DataFrame(X[100:], columns=feature_names)
    df_current['target'] = Y[100:]
    
    # Run Evidently Report
    data_report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset()
    ])
    res = data_report.run(reference_data=df_reference, current_data=df_current)
    
    # Save the report as an HTML artifact for production CI/CD visibility
    report_path = "data_validation_report.html"
    res.save_html(report_path)
    print(f"\n[INFO] Saved data validation report to {report_path}")
    
    # Extract metrics to assert no critical drift/failures
    report_dict = res.dict()
    # Find the drifted columns count metric config type
    drift_metric = next(
        m for m in report_dict["metrics"] 
        if m.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount"
    )
    drift_share = drift_metric["value"]["share"]
    assert drift_share < 0.3, f"Critical data drift detected! Drifted columns share: {drift_share:.2%}"


# ==========================================
# TEST CATEGORY 2: Model Performance & Accuracy Checks
# ==========================================

def test_model_accuracy(trained_model, eval_dataset):
    """Ensure the trained model holds its baseline classification accuracy."""
    X_np, Y_np = eval_dataset
    X = torch.tensor(X_np)
    Y = torch.tensor(Y_np)
    
    with torch.no_grad():
        preds = trained_model(X)
        preds_bin = (preds > 0.5).float()
        accuracy = (preds_bin == Y).float().mean().item()
        
    print(f"\n[INFO] Model Accuracy: {accuracy:.4%}")
    assert accuracy >= 0.85, f"Model accuracy too low: {accuracy:.2%}"


# ==========================================
# TEST CATEGORY 3: Model Robustness & Fuzzing (Perturbation Testing)
# ==========================================

def test_model_robustness_fuzzing(trained_model, eval_dataset):
    """ML Fuzzing test: Add noise to inputs, assert predictions remain stable."""
    X_np, _ = eval_dataset
    X = torch.tensor(X_np)
    
    with torch.no_grad():
        original_preds = (trained_model(X) > 0.5).float()
        
        # Inject Gaussian perturbation (standard deviation of 0.05)
        noise = torch.randn_like(X) * 0.05
        fuzzed_X = X + noise
        
        fuzzed_preds = (trained_model(fuzzed_X) > 0.5).float()
        
        # Calculate consistency share
        consistency = (original_preds == fuzzed_preds).float().mean().item()
        
    print(f"\n[INFO] Model Fuzzing Prediction Consistency: {consistency:.2%}")
    assert consistency >= 0.90, f"Model is not robust to noise! Prediction consistency fell to {consistency:.2%}"


# ==========================================
# TEST CATEGORY 4: Boundary & Safety Checks
# ==========================================

def test_model_boundary_safety(trained_model):
    """Ensure prediction probabilities are always safely bounded (0.0 to 1.0)."""
    # Extremely large values, extremely small values, and all zeros
    extreme_inputs = torch.tensor([
        [1e6] * 10,
        [-1e6] * 10,
        [0.0] * 10
    ], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = trained_model(extreme_inputs)
        
    for i, output in enumerate(outputs):
        val = output.item()
        assert 0.0 <= val <= 1.0, f"Safety violation! Prediction {val} out of bounds [0.0, 1.0] for input index {i}"
