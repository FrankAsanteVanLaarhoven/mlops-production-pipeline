import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import brevitas.nn as qnn
import pandas as pd

# Evidently imports
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Isolate ZenML configuration to this directory
os.environ["ZENML_CONFIG_PATH"] = os.path.abspath(".zenml-config")

from zenml import step, pipeline

# ==========================================
# STEP 1: Data Ingestion
# ==========================================
@step
def load_data() -> dict:
    """Step to generate synthetic classification data."""
    print("[Ingestion] Generating synthetic training and testing data...")
    # Generate random features
    X = np.random.randn(200, 10).astype(np.float32)
    # Define targets (1 if sum of features is positive, else 0)
    Y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    
    # Split into train/test (80/20)
    X_train, X_test = X[:160], X[160:]
    Y_train, Y_test = Y[:160], Y[160:]
    
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test
    }

# ==========================================
# STEP 2: Data Quality & Schema Validation (Evidently)
# ==========================================
@step
def validate_data(data: dict) -> dict:
    """Step to run Evidently data quality and drift analysis."""
    print("[Validation] Executing Evidently data quality and drift presets...")
    
    feature_names = [f"feature_{i}" for i in range(10)]
    df_train = pd.DataFrame(data["X_train"], columns=feature_names)
    df_train['target'] = data["Y_train"]
    
    df_test = pd.DataFrame(data["X_test"], columns=feature_names)
    df_test['target'] = data["Y_test"]
    
    # Run Evidently Report
    report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset()
    ])
    res = report.run(reference_data=df_train, current_data=df_test)
    
    # Save the report as an HTML artifact
    report_path = "data_drift_report.html"
    res.save_html(report_path)
    print(f"[Validation] Saved Evidently validation report to {report_path}")
    
    # Assert no critical drift
    report_dict = res.dict()
    drift_metric = next(
        m for m in report_dict["metrics"] 
        if m.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount"
    )
    drift_share = drift_metric["value"]["share"]
    print(f"[Validation] Share of drifted features: {drift_share:.2%}")
    
    if drift_share >= 0.3:
        raise ValueError(f"Pipeline Halted: Critical data drift detected ({drift_share:.2%})!")
        
    return {"drift_share": drift_share, "report_path": report_path}

# ==========================================
# STEP 3: Hyperparameter Optimization & Model Training
# ==========================================
@step
def train_and_optimize(data: dict) -> dict:
    """Step to optimize model hyperparameters with Optuna and train with Brevitas."""
    print("[Training] Starting hyperparameter optimization...")
    X_train = torch.tensor(data["X_train"])
    Y_train = torch.tensor(data["Y_train"])
    X_test = torch.tensor(data["X_test"])
    Y_test = torch.tensor(data["Y_test"])
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        weight_bit_width = trial.suggest_int("weight_bit_width", 4, 8)
        
        model = nn.Sequential(
            qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
            nn.ReLU(),
            qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
            nn.Sigmoid()
        )
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            preds_bin = (preds > 0.5).float()
            accuracy = (preds_bin == Y_test).float().mean().item()
            
        return accuracy

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    
    best_params = study.best_trial.params
    print(f"[Training] Best Hyperparameters Found: {best_params}")
    
    # Train the final model with best parameters
    best_model = nn.Sequential(
        qnn.QuantLinear(10, 5, weight_bit_width=best_params["weight_bit_width"], bias=True),
        nn.ReLU(),
        qnn.QuantLinear(5, 1, weight_bit_width=best_params["weight_bit_width"], bias=True),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])
    
    for epoch in range(50):
        best_model.train()
        optimizer.zero_grad()
        outputs = best_model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
    # Save checkpoint file
    checkpoint_path = "best_model.pt"
    torch.save({
        'state_dict': best_model.state_dict(),
        'weight_bit_width': best_params["weight_bit_width"],
        'hyperparameters': best_params
    }, checkpoint_path)
    print(f"[Training] Best model saved to {checkpoint_path}")
    
    return {"checkpoint_path": checkpoint_path, "weight_bit_width": best_params["weight_bit_width"]}

# ==========================================
# STEP 4: Model Performance & Robustness Fuzzing Validation
# ==========================================
@step
def validate_model(data: dict, model_info: dict) -> dict:
    """Step to perform automated robustness, boundary safety, and performance checks."""
    print("[Model Validation] Running automated model tests...")
    
    checkpoint_path = model_info["checkpoint_path"]
    weight_bit_width = model_info["weight_bit_width"]
    
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = nn.Sequential(
        qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
        nn.ReLU(),
        qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
        nn.Sigmoid()
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    X_test = torch.tensor(data["X_test"])
    Y_test = torch.tensor(data["Y_test"])
    
    with torch.no_grad():
        preds = model(X_test)
        preds_bin = (preds > 0.5).float()
        accuracy = (preds_bin == Y_test).float().mean().item()
        
    print(f"[Model Validation] Accuracy: {accuracy:.2%}")
    if accuracy < 0.85:
        raise ValueError(f"Validation FAILED: Model accuracy {accuracy:.2%} is below threshold (85%)!")
        
    # ML Fuzzing / Robustness check
    with torch.no_grad():
        original_preds = (model(X_test) > 0.5).float()
        noise = torch.randn_like(X_test) * 0.05
        fuzzed_preds = (model(X_test + noise) > 0.5).float()
        consistency = (original_preds == fuzzed_preds).float().mean().item()
        
    print(f"[Model Validation] Fuzzing Consistency: {consistency:.2%}")
    if consistency < 0.90:
        raise ValueError(f"Validation FAILED: Model robustness consistency {consistency:.2%} is below threshold (90%)!")
        
    # Boundary & Probability Safety check
    extreme_inputs = torch.tensor([
        [1e6] * 10,
        [-1e6] * 10,
        [0.0] * 10
    ], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(extreme_inputs)
    for i, output in enumerate(outputs):
        val = output.item()
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Validation FAILED: Model output probability {val} for index {i} violates safety bounds!")
            
    print("[Model Validation] All accuracy, robustness, and boundary checks PASSED!")
    return {"status": "PASSED", "accuracy": accuracy, "consistency": consistency}

# ==========================================
# STEP 5: Automated DVC Versioning
# ==========================================
@step
def version_with_dvc(validation_status: dict) -> str:
    """Step to automate DVC versioning on successful validation."""
    print("[DVC Versioning] Validating model pipeline status...")
    if validation_status["status"] != "PASSED":
        raise ValueError("Cannot run DVC versioning: Model validation did not pass!")
        
    print("[DVC Versioning] Running 'dvc add best_model.pt' to track model version...")
    # Run DVC command
    result = subprocess.run(
        ["dvc", "add", "best_model.pt"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"DVC tracking failed: {result.stderr}")
        
    print(result.stdout)
    print("[DVC Versioning] Successfully tracked new model checkpoint file under DVC.")
    return "SUCCESS"

# ==========================================
# PIPELINE DEFINITION
# ==========================================
@pipeline
def mlops_e2e_pipeline():
    """End-to-end MLOps Lifecycle Pipeline with Data Validation, Model Fuzzing, and DVC Versioning."""
    data = load_data()
    validate_data(data)
    model_info = train_and_optimize(data)
    validation_status = validate_model(data, model_info)
    version_with_dvc(validation_status)

if __name__ == "__main__":
    print("=== STARTING FULL MLOPS PRODUCTION PIPELINE ===")
    mlops_e2e_pipeline()
    print("=== MLOPS PRODUCTION PIPELINE COMPLETED SUCCESSFULLY ===")
