"""Ray Serve model serving adapter implementation."""

import json
import os

import ray
import torch
from pydantic import ValidationError
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..core.serving import BaseServingService
from ..registry import load_latest
from ..schemas import PredictionRequest, PredictionResponse, validate_feature_vector

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
        return JSONResponse({"error": f"no route for {request.method} {path}"}, status_code=404)

    async def _predict(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "request body is not valid JSON"}, status_code=400)

        try:
            parsed = PredictionRequest.model_validate(payload)
            validate_feature_vector(parsed.features, self.n_features, self.max_abs_feature_value)
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


class RayServingAdapter(BaseServingService):
    """Ray Serve implementation of BaseServingService."""

    def start(self, host: str, port: int, registry_root: str, max_abs_feature_value: float) -> None:
        """Start the Ray Serve deployment serving the latest model."""
        # Replicas run from Ray's packaged copy of the working dir; resolve the
        # registry to an absolute path so they read the real one.
        resolved_root = str(os.path.abspath(registry_root))
        ray.init(logging_level="warning")
        serve.start(http_options={"host": host, "port": port})
        serve.run(
            ModelService.bind(resolved_root, max_abs_feature_value),
            route_prefix="/",
        )
        print(f"[serving] live on http://{host}:{port}")

    def stop(self) -> None:
        """Stop the Ray Serve deployment and shut down Ray."""
        print("[serving] shutting down Ray Serve")
        serve.shutdown()
        ray.shutdown()
