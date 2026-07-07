import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
import torch.nn as nn
import brevitas.nn as qnn
import ray
from ray import serve
import requests
import time
from starlette.requests import Request
from starlette.responses import JSONResponse

# Define the model builder
def build_model(weight_bit_width):
    return nn.Sequential(
        qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
        nn.ReLU(),
        qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
        nn.Sigmoid()
    )

# Define the Ray Serve Deployment
@serve.deployment
class QuantModelDeployment:
    def __init__(self):
        checkpoint_path = "best_model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
            
        checkpoint = torch.load(checkpoint_path)
        weight_bit_width = checkpoint['weight_bit_width']
        
        self.model = build_model(weight_bit_width)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"[Deployment] Loaded Brevitas model with {weight_bit_width}-bit weights successfully.")

    async def __call__(self, request: Request) -> JSONResponse:
        try:
            json_data = await request.json()
            features = json_data.get("features")
            
            if not features or len(features) != 10:
                return JSONResponse(
                    {"error": "Please provide a list of exactly 10 float features under 'features'."},
                    status_code=400
                )
            
            # Run model inference
            input_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                prob = self.model(input_tensor).item()
                prediction = 1 if prob > 0.5 else 0
                
            return JSONResponse({
                "predicted_class": prediction,
                "probability": prob
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

def main():
    print("Initializing Ray and Serve...")
    # Initialize Ray locally with runtime_env to pass PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION to all workers
    ray.init(
        logging_level="warning",
        runtime_env={"env_vars": {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}}
    )
    # Start Serve on port 8000
    serve.start(http_options={"port": 8000})
    
    print("Deploying the quantized Brevitas model via Ray Serve...")
    serve.run(QuantModelDeployment.bind())
    print("Deployment is active on http://localhost:8000/")
    
    # Send test request
    print("\nSending a test HTTP POST request to the served model...")
    # Features sum to positive -> expect class 1
    test_features = [1.2, 0.5, 0.8, -0.2, 0.4, 1.1, -0.5, 0.9, -0.1, 0.3]
    payload = {"features": test_features}
    
    try:
        response = requests.post("http://localhost:8000/", json=payload)
        print("Response received from Ray Serve API:")
        print(f"  Status Code: {response.status_code}")
        print(f"  JSON Body: {response.json()}")
    except Exception as e:
        print(f"  Request failed: {e}")
        
    print("\nCleaning up and shutting down Ray Serve...")
    serve.shutdown()
    ray.shutdown()
    print("Ray Serve shutdown complete.")

if __name__ == "__main__":
    main()
