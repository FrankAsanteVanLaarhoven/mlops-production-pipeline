import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import brevitas.nn as qnn

# Isolate ZenML configuration to this directory
os.environ["ZENML_CONFIG_PATH"] = os.path.abspath(".zenml-config")

from zenml import step, pipeline

@step
def load_data() -> dict:
    """Step to generate synthetic classification data and return as a dictionary of NumPy arrays."""
    print("Step 1: Generating synthetic data...")
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

@step
def optimize_and_train(data: dict) -> dict:
    """Step to optimize model hyperparameters with Optuna and train with Brevitas."""
    print("Step 2: Training and optimizing quantized model...")
    # Convert numpy arrays to torch tensors
    X_train = torch.tensor(data["X_train"])
    Y_train = torch.tensor(data["Y_train"])
    X_test = torch.tensor(data["X_test"])
    Y_test = torch.tensor(data["Y_test"])
    
    def objective(trial):
        # Hyperparameters to optimize
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        weight_bit_width = trial.suggest_int("weight_bit_width", 4, 8)
        
        # Build Brevitas Quantized model
        model = nn.Sequential(
            qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
            nn.ReLU(),
            qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
            nn.Sigmoid()
        )
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train for 20 epochs
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
            
        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            preds_bin = (preds > 0.5).float()
            accuracy = (preds_bin == Y_test).float().mean().item()
            
        return accuracy

    print("Starting Optuna study (running 5 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    
    best_trial = study.best_trial
    print(f"Optimization complete!")
    print(f"Best trial value (Accuracy): {best_trial.value:.4f}")
    print(f"Best hyperparameters: {best_trial.params}")
    
    return best_trial.params

@pipeline
def mlops_training_pipeline():
    """ZenML pipeline containing the loading and training steps."""
    data = load_data()
    optimize_and_train(data)

if __name__ == "__main__":
    print("Initializing ZenML pipeline run...")
    mlops_training_pipeline()
    print("ZenML pipeline run completed successfully.")
