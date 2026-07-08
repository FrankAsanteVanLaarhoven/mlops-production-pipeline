import os
import subprocess
import time
import httpx
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mlops_pipeline.config import PipelineConfig

app = FastAPI(title="MLOps Production Pipeline Dashboard")

# Root directory of the project
ROOT_DIR = Path(__file__).parent.resolve()
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"
STATIC_DIR = ROOT_DIR / "dashboard_static"

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Keep track of active training process
training_process = None
train_log_file = ROOT_DIR / "artifacts" / "train_run.log"

class FeatureInput(BaseModel):
    features: list[float]

@app.get("/api/status")
async def get_system_status():
    """Check if the Ray Serve server is listening on port 8000, falling back to local registry or default model."""
    config_path = ROOT_DIR / "configs" / "pipeline.yaml"
    cfg = PipelineConfig.from_yaml(config_path) if config_path.exists() else PipelineConfig()
    gates_config = cfg.gates.model_dump(mode="json")

    # 1. Try to connect to Ray Serve
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:8000/health", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "serve_status": "online",
                    "model_version": data.get("model_version", "unknown"),
                    "metrics": data.get("metrics", {}),
                    "gates_config": gates_config,
                }
    except Exception:
        pass

    # 2. Ray is offline. Fall back to local registry / default model (serverless mode)
    try:
        from mlops_pipeline.registry import load_latest
        registry_root = ROOT_DIR / "artifacts" / "registry"
        default_model_dir = ROOT_DIR / "src" / "mlops_pipeline" / "default_model"

        if (registry_root / "latest.json").exists():
            _, card = load_latest(registry_root)
        elif (default_model_dir / "latest.json").exists():
            _, card = load_latest(default_model_dir)
        else:
            card = None

        if card is not None:
            return {
                "serve_status": "online",
                "model_version": card.get("version", "unknown"),
                "metrics": card.get("metrics", {}),
                "gates_config": gates_config,
            }
    except Exception as e:
        print(f"[dashboard] failed to load local model status: {e}")

    return {
        "serve_status": "offline",
        "model_version": "None",
        "metrics": {},
        "gates_config": gates_config,
    }


@app.post("/api/predict")
async def proxy_predict(payload: FeatureInput):
    """Proxy prediction requests to Ray Serve, falling back to direct local inference if Ray is offline."""
    # 1. Try Ray Serve
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://127.0.0.1:8000/predict",
                json={"features": payload.features},
                timeout=5.0
            )
            if resp.status_code == 200:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception:
        pass

    # 2. Fallback to direct local inference using PyTorch & Brevitas
    try:
        from mlops_pipeline.registry import load_latest
        import torch
        
        registry_root = ROOT_DIR / "artifacts" / "registry"
        default_model_dir = ROOT_DIR / "src" / "mlops_pipeline" / "default_model"

        if (registry_root / "latest.json").exists():
            model, card = load_latest(registry_root)
        elif (default_model_dir / "latest.json").exists():
            model, card = load_latest(default_model_dir)
        else:
            raise FileNotFoundError("No model found in registry or default_model")

        # Validate feature dimension
        n_features = card.get("architecture", {}).get("n_features", 10)
        if len(payload.features) != n_features:
            return JSONResponse(
                content={"error": "input validation failed", "details": f"expected {n_features} features, got {len(payload.features)}"},
                status_code=400
            )

        # Validate feature value bounds
        config_path = ROOT_DIR / "configs" / "pipeline.yaml"
        cfg = PipelineConfig.from_yaml(config_path) if config_path.exists() else PipelineConfig()
        max_abs = cfg.serving.max_abs_feature_value
        if any(abs(x) > max_abs for x in payload.features):
            return JSONResponse(
                content={"error": "input validation failed", "details": f"out-of-distribution (|value| > {max_abs})"},
                status_code=400
            )

        # Execute model forward pass
        model.eval()
        with torch.no_grad():
            x = torch.tensor([payload.features], dtype=torch.float32)
            prob = float(model(x).item())
            pred_class = 1 if prob > 0.5 else 0

        # Log prediction to prod log if filesystem is writable
        prod_data_path = ROOT_DIR / "artifacts" / "serving" / "production_data.csv"
        try:
            prod_data_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = prod_data_path.exists()
            with open(prod_data_path, "a") as f:
                if not file_exists:
                    f.write(",".join([f"feature_{i}" for i in range(len(payload.features))]) + "\n")
                f.write(",".join(map(str, payload.features)) + "\n")
        except Exception:
            pass

        return {
            "predicted_class": pred_class,
            "probability": prob,
            "model_version": card["version"]
        }
    except Exception as e:
        return JSONResponse(
            content={"error": "inference failed", "details": str(e)},
            status_code=502
        )


@app.post("/api/train")
async def trigger_training():
    """Launch the ZenML training pipeline in the background (blocked on Vercel)."""
    global training_process
    
    # Block running training on Vercel serverless platform
    if "VERCEL" in os.environ:
        return JSONResponse(
            content={
                "status": "error",
                "message": "Training pipeline cannot be executed in serverless environments (read-only filesystem, resource constraints). Please run training locally using 'make train'."
            },
            status_code=400
        )

    # Check if already training
    if training_process is not None and training_process.poll() is None:
        return JSONResponse(content={"status": "error", "message": "Training is already in progress"}, status_code=400)
    
    # Reset log file
    if train_log_file.exists():
        train_log_file.unlink()
        
    # Launch subprocess
    try:
        # We run 'make train' (uv run mlops-train)
        training_process = subprocess.Popen(
            ["make", "train"],
            cwd=str(ROOT_DIR),
            stdout=open(train_log_file, "w"),
            stderr=subprocess.STDOUT,
            text=True
        )
        return {"status": "started", "pid": training_process.pid}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/train/status")
async def get_training_status():
    """Return whether training is currently running."""
    global training_process
    if training_process is not None and training_process.poll() is None:
        return {"status": "running"}
    return {"status": "idle"}

@app.get("/api/train/logs")
async def stream_training_logs():
    """Stream stdout logs of the active training run using SSE."""
    async def log_generator():
        # Wait until log file is created
        attempts = 0
        while not train_log_file.exists() and attempts < 20:
            await asyncio.sleep(0.2)
            attempts += 1
            
        if not train_log_file.exists():
            yield "data: Error: Log file not created.\n\n"
            return
            
        with open(train_log_file, "r") as f:
            while True:
                line = f.readline()
                if line:
                    # Clean color codes or special chars if any, and send
                    yield f"data: {line}"
                else:
                    # Check if process ended
                    global training_process
                    if training_process is not None and training_process.poll() is not None:
                        # Process finished, read remaining lines
                        remaining = f.read()
                        if remaining:
                            for rem_line in remaining.splitlines(keepends=True):
                                yield f"data: {rem_line}"
                        yield "data: === TRAINING PROCESS CONCLUDED ===\n\n"
                        break
                    await asyncio.sleep(0.5)
                    
    import asyncio
    return StreamingResponse(log_generator(), media_type="text/event-stream")

# Serve Evidently reports statically under /reports/
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# Serve frontend static files
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start dashboard on port 8080
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8080, reload=True)
