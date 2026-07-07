import os
import requests
import torch
import torch.nn as nn
import brevitas.nn as qnn
import ray
from ray import serve
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from guardrails import Guard

# Isolate ZenML and Ray environment settings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 1. Define Pydantic structures for Guardrails validation
class ModelInputSchema(BaseModel):
    features: list[float] = Field(..., min_items=10, max_items=10, description="Input features (must be exactly 10 values)")

    @field_validator("features")
    @classmethod
    def check_sensible_ranges(cls, v):
        # Fuzzy Labs Safety Gate: Warn or reject inputs that are extreme out-of-distribution values
        for val in v:
            if abs(val) > 10.0:
                raise ValueError(f"Feature value {val} is an extreme out-of-distribution anomaly (>10.0)!")
        return v

class ModelOutputSchema(BaseModel):
    predicted_class: int = Field(..., description="The predicted binary class (must be 0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Inference confidence score bounded in [0, 1]")

# 2. Re-create the quantized model structure
def build_model(weight_bit_width):
    return nn.Sequential(
        qnn.QuantLinear(10, 5, weight_bit_width=weight_bit_width, bias=True),
        nn.ReLU(),
        qnn.QuantLinear(5, 1, weight_bit_width=weight_bit_width, bias=True),
        nn.Sigmoid()
    )

# 3. Create Ray Serve Deployment wrapped with Guardrails AI
@serve.deployment
class GuardedModelDeployment:
    def __init__(self):
        checkpoint_path = "/Users/favl/.gemini/antigravity-ide/scratch/mlops-env/best_model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint '{checkpoint_path}' not found! Run training pipeline first.")
        
        checkpoint = torch.load(checkpoint_path)
        weight_bit_width = checkpoint['weight_bit_width']
        
        self.model = build_model(weight_bit_width)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Initialize Guardrails AI Guards
        self.input_guard = Guard.for_pydantic(ModelInputSchema)
        self.output_guard = Guard.for_pydantic(ModelOutputSchema)
        print("[Deployment] Guarded deployment initialized with Input/Output validation schemas.")

    async def __call__(self, request) -> JSONResponse:
        try:
            import json
            # Parse raw HTTP request body
            raw_data = await request.json()
            
            # Step A: Validate input using Pydantic and Guardrails
            print(f"[Schema] Validating input request payload with Pydantic...")
            try:
                validated_input = ModelInputSchema(**raw_data)
            except Exception as pydantic_err:
                return JSONResponse({
                    "error": "Input Schema Validation Failure",
                    "details": str(pydantic_err)
                }, status_code=400)
            
            validation_result = self.input_guard.parse(json.dumps(raw_data))
            # Check if validation succeeded
            if not validation_result.validation_passed:
                return JSONResponse({
                    "error": "Input Guardrail Failure",
                    "details": validation_result.error
                }, status_code=400)
            
            features = raw_data["features"]
            
            # Step B: Model Inference
            input_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                prob_tensor = self.model(input_tensor)
                prob = prob_tensor.item()
                predicted_class = 1 if prob > 0.5 else 0

            # Step C: Validate output using Pydantic and Guardrails
            inference_output = {
                "predicted_class": predicted_class,
                "probability": prob
            }
            print(f"[Schema] Validating model inference output with Pydantic...")
            try:
                ModelOutputSchema(**inference_output)
            except Exception as pydantic_err:
                return JSONResponse({
                    "error": "Output Schema Validation Failure",
                    "details": str(pydantic_err)
                }, status_code=500)
                
            print(f"[Guardrails] Validating model inference output: {inference_output}")
            output_validation = self.output_guard.parse(json.dumps(inference_output))
            
            if not output_validation.validation_passed:
                return JSONResponse({
                    "error": "Output Guardrail Failure",
                    "details": output_validation.error
                }, status_code=500)

            return JSONResponse(inference_output)

        except Exception as e:
            return JSONResponse({"error": f"Exception encountered: {str(e)}"}, status_code=500)

def main():
    print("Initializing Ray cluster and starting Serve...")
    ray.init(
        logging_level="warning",
        runtime_env={"env_vars": {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}}
    )
    serve.start(http_options={"port": 8000})
    
    print("Deploying Guarded deployment via Ray Serve...")
    serve.run(GuardedModelDeployment.bind())
    print("Deployment is active on http://localhost:8000/")
    
    # 4. Run verification queries demonstrating Guardrails in action
    
    # Query 1: Valid request
    print("\n--- QUERY 1: Sending Valid Request ---")
    valid_payload = {"features": [0.5, -0.2, 0.1, 0.4, 0.0, -0.1, 0.3, 0.2, -0.4, 0.8]}
    resp = requests.post("http://localhost:8000/", json=valid_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    
    # Query 2: Invalid Request (length mismatch)
    print("\n--- QUERY 2: Sending Malformed Request (Wrong Feature Length) ---")
    invalid_len_payload = {"features": [0.5, -0.2, 0.1]}
    resp = requests.post("http://localhost:8000/", json=invalid_len_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")

    # Query 3: Invalid Request (out of distribution values)
    print("\n--- QUERY 3: Sending Anomalous Request (Out of range value) ---")
    anomaly_payload = {"features": [15.5, -0.2, 0.1, 0.4, 0.0, -0.1, 0.3, 0.2, -0.4, 0.8]}
    resp = requests.post("http://localhost:8000/", json=anomaly_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")

    print("\nShutting down Ray Serve...")
    serve.shutdown()
    ray.shutdown()
    print("Ray cluster shutdown complete.")

if __name__ == "__main__":
    main()
