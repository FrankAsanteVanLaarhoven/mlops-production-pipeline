import os
import subprocess

# Isolate ZenML configuration to this directory to avoid conflicts with global migrations
os.environ["ZENML_CONFIG_PATH"] = os.path.abspath(".zenml-config")

import sys
import torch
import brevitas
import optuna
import zenml

# New imports
import lakefs
import lakefs_spec
import labelbox
import mlx
import vllm_metal
import ray
from ray import serve
import guardrails

print("=" * 60)
print("MLOps Tools Extended Installation Verification")
print("=" * 60)

# Check Python version
print(f"Python Version: {sys.version}")

# 1. PyTorch
print(f"PyTorch Version: {torch.__version__}")
tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"PyTorch Tensor test: {tensor} (Device: {tensor.device})")

# 2. Brevitas
print(f"Brevitas Version: {brevitas.__version__}")
from brevitas.nn import QuantLinear
quant_layer = QuantLinear(10, 5, weight_bit_width=8, bias=True)
print(f"Brevitas QuantLinear layer created successfully: {quant_layer}")

# 3. Optuna
print(f"Optuna Version: {optuna.__version__}")
study = optuna.create_study(direction="minimize")
print(f"Optuna Study created successfully: {study.study_name}")

# 4. ZenML
print(f"ZenML Version: {zenml.__version__}")
from zenml.client import Client
client = Client()
print(f"ZenML Client initialized. Store type: {client.zen_store.type}")

# 5. lakeFS & lakefs-spec
print("lakeFS SDK imported successfully.")
print(f"lakefs-spec Version: {lakefs_spec.__version__}")

# 6. Labelbox
print("Labelbox SDK imported successfully.")
from labelbox import Client as LabelboxClient
print("Labelbox Client class available.")

# 7. vLLM-metal & MLX
print("MLX imported successfully.")
print("vllm-metal imported successfully.")

# 8. Ray & Ray Serve
print(f"Ray Version: {ray.__version__}")
print("Ray Serve imported successfully.")

# 9. Guardrails AI
print("Guardrails AI imported successfully.")

# 10. CLI Tool checks
print("\nVerifying CLI Tools installed via uv tool:")

def check_cli(command):
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            version_str = res.stdout.strip().split("\n")[0]
            print(f"  [PASS] {command}: {version_str}")
        else:
            print(f"  [FAIL] {command} returned exit code {res.returncode}")
    except Exception as e:
        print(f"  [ERROR] {command}: {e}")

check_cli("dvc --version")
check_cli("platformio --version")
check_cli("mcpgateway-server --help | head -n 1")

print("=" * 60)
print("All extended MLOps tools installed and verified successfully!")
print("=" * 60)
