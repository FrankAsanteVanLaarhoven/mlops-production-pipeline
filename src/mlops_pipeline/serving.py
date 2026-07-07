"""Guarded model serving on Ray Serve.

The deployment loads whatever the registry's `latest` pointer selects, validates
every request and response against the contracts in schemas.py, and exposes:

    GET  /health   → status + model version + registered metrics
    POST /predict  → guarded inference (also served at POST /)

Guardrails AI wraps the response contract when installed; the Pydantic contract
is always enforced regardless.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ray
import torch
from pydantic import ValidationError
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import PipelineConfig
from .registry import load_latest
from .schemas import PredictionRequest, PredictionResponse, validate_feature_vector

try:
    from guardrails import Guard

    _HAS_GUARDRAILS = True
except ImportError:
    _HAS_GUARDRAILS = False


@serve.deployment
class ModelService:
    """Serves the registry's latest model behind request/response contracts."""

    def __init__(self, registry_root: str, max_abs_feature_value: float):
        """Load the model selected by the registry's latest pointer."""
        self.model, self.card = load_latest(registry_root)
        self.n_features = int(self.card["architecture"]["n_features"])
        self.max_abs_feature_value = max_abs_feature_value
        self.output_guard = Guard.for_pydantic(PredictionResponse) if _HAS_GUARDRAILS else None
        print(
            f"[serving] loaded {self.card['version']} from {registry_root} "
            f"(guardrails={'on' if self.output_guard else 'off'})"
        )

    async def __call__(self, request: Request) -> JSONResponse:
        """Route health and prediction requests."""
        path = request.url.path
        if request.method == "GET" and path in ("/", "/health"):
            return JSONResponse(
                {
                    "status": "ok",
                    "model_version": self.card["version"],
                    "metrics": self.card["metrics"],
                }
            )
        if request.method == "POST" and path in ("/", "/predict"):
            return await self._predict(request)
        return JSONResponse(
            {"error": f"no route for {request.method} {path}"}, status_code=404
        )

    async def _predict(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "request body is not valid JSON"}, status_code=400)

        try:
            parsed = PredictionRequest.model_validate(payload)
            validate_feature_vector(
                parsed.features, self.n_features, self.max_abs_feature_value
            )
        except (ValidationError, ValueError) as err:
            return JSONResponse(
                {"error": "input validation failed", "details": str(err)}, status_code=400
            )

        with torch.no_grad():
            probability = float(
                self.model(torch.tensor([parsed.features], dtype=torch.float32)).item()
            )
        response = {
            "predicted_class": int(probability > 0.5),
            "probability": probability,
            "model_version": self.card["version"],
        }

        try:
            PredictionResponse.model_validate(response)
            if self.output_guard is not None:
                outcome = self.output_guard.parse(json.dumps(response))
                if not outcome.validation_passed:
                    raise ValueError(str(outcome.error))
        except (ValidationError, ValueError) as err:
            return JSONResponse(
                {"error": "output validation failed", "details": str(err)}, status_code=500
            )
        return JSONResponse(response)


def _run_smoke_test(base_url: str) -> int:
    """Exercise the guarded API; return a process exit code."""
    import requests

    checks = [
        ("GET /health", "get", "/health", None, 200),
        (
            "valid request",
            "post",
            "/predict",
            {"features": [0.5, -0.2, 0.1, 0.4, 0.0, -0.1, 0.3, 0.2, -0.4, 0.8]},
            200,
        ),
        ("wrong feature count", "post", "/predict", {"features": [0.5, -0.2, 0.1]}, 400),
        (
            "out-of-distribution value",
            "post",
            "/predict",
            {"features": [15.5, -0.2, 0.1, 0.4, 0.0, -0.1, 0.3, 0.2, -0.4, 0.8]},
            400,
        ),
    ]
    failures = 0
    for name, method, path, body, expected_status in checks:
        resp = getattr(requests, method)(base_url + path, json=body, timeout=30)
        ok = resp.status_code == expected_status
        failures += 0 if ok else 1
        print(
            f"[smoke] {'PASS' if ok else 'FAIL'} {name}: "
            f"status={resp.status_code} (expected {expected_status}) body={resp.json()}"
        )
    print(f"[smoke] {len(checks) - failures}/{len(checks)} checks passed")
    return 0 if failures == 0 else 1


def main() -> None:
    """CLI entry point for `mlops-serve`."""
    parser = argparse.ArgumentParser(description="Serve the latest registered model")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--registry-root",
        default=None,
        help="override the registry root (also honours MODEL_REGISTRY_ROOT)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run guarded-API smoke checks against the deployment, then exit",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    registry_root = (
        args.registry_root
        or os.environ.get("MODEL_REGISTRY_ROOT")
        or str(cfg.registry.root)
    )
    # Replicas run from Ray's packaged copy of the working dir; resolve the
    # registry to an absolute path so they read the real one.
    registry_root = str(Path(registry_root).resolve())

    ray.init(logging_level="warning")
    serve.start(http_options={"host": cfg.serving.host, "port": cfg.serving.port})
    serve.run(
        ModelService.bind(registry_root, cfg.serving.max_abs_feature_value),
        route_prefix="/",
    )
    print(f"[serving] live on http://{cfg.serving.host}:{cfg.serving.port}")

    if args.smoke_test:
        exit_code = _run_smoke_test(f"http://127.0.0.1:{cfg.serving.port}")
        serve.shutdown()
        ray.shutdown()
        sys.exit(exit_code)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[serving] shutting down")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
