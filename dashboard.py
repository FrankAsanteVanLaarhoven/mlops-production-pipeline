import os
import subprocess
import time
import httpx
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    """Check if the Ray Serve server is listening on port 8000."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:8000/health", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "serve_status": "online",
                    "model_version": data.get("model_version", "unknown"),
                    "metrics": data.get("metrics", {})
                }
    except Exception:
        pass
    
    return {
        "serve_status": "offline",
        "model_version": "None",
        "metrics": {}
    }

@app.post("/api/predict")
async def proxy_predict(payload: FeatureInput):
    """Proxy prediction requests to Ray Serve on port 8000."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://127.0.0.1:8000/predict",
                json={"features": payload.features},
                timeout=5.0
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": "failed to connect to Ray Serve", "details": str(e)},
            status_code=502
        )

@app.post("/api/train")
async def trigger_training():
    """Launch the ZenML training pipeline in the background."""
    global training_process
    
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
