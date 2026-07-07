import os
import torch
import torch.nn as nn
import torch.optim as optim
import brevitas.nn as qnn
import numpy as np

def train_and_save():
    print("Generating training data...")
    # Generate 1000 samples for better training
    X_train = torch.randn(800, 10)
    Y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)
    
    # Best parameters from Optuna run
    lr = 0.04
    weight_bit_width = 6
    
    print(f"Building Brevitas quantized model (bit-width: {weight_bit_width})...")
    model = nn.Sequential(
        qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
        nn.ReLU(),
        qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Training model for 50 epochs...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
            
    # Save the model state dictionary and structure metadata
    model_path = "best_model.pt"
    torch.save({
        'state_dict': model.state_dict(),
        'weight_bit_width': weight_bit_width
    }, model_path)
    print(f"Model saved successfully to '{model_path}'")

if __name__ == "__main__":
    train_and_save()
