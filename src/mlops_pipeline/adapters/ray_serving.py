"""Ray Serve model serving adapter implementation."""

import json
import os
from pathlib import Path

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

    def __init__(
        self,
        registry_root: str,
        max_abs_feature_value: float,
        production_data_path: str | None = None,
    ):
        """Load the model selected by the registry's latest pointer."""
        self.model, self.card = load_latest(registry_root)
        self.n_features = int(self.card["architecture"]["n_features"])
        self.max_abs_feature_value = max_abs_feature_value
        self.production_data_path = Path(production_data_path) if production_data_path else None
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

        if self.production_data_path is not None:
            self._log_prediction(parsed.features)

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

    def _log_prediction(self, features: list[float]) -> None:
        """Log production features to a CSV file."""
        try:
            self.production_data_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.production_data_path.exists()
            with open(self.production_data_path, "a") as f:
                if not file_exists:
                    feature_names = self.card.get("feature_names")
                    if not feature_names:
                        feature_names = [f"feature_{i}" for i in range(self.n_features)]
                    f.write(",".join(feature_names) + "\n")
                f.write(",".join(map(str, features)) + "\n")
        except Exception as e:
            print(f"[serving] failed to log prediction: {e}")


class RayServingAdapter(BaseServingService):
    """Ray Serve implementation of BaseServingService."""

    def start(
        self,
        host: str,
        port: int,
        registry_root: str,
        max_abs_feature_value: float,
        production_data_path: str | None = None,
    ) -> None:
        """Start the Ray Serve deployment serving the latest model."""
        # Replicas run from Ray's packaged copy of the working dir; resolve the
        # registry to an absolute path so they read the real one.
        resolved_root = str(os.path.abspath(registry_root))
        ray.init(logging_level="warning")
        serve.start(http_options={"host": host, "port": port})
        serve.run(
            ModelService.bind(resolved_root, max_abs_feature_value, production_data_path),
            route_prefix="/",
        )
        print(f"[serving] live on http://{host}:{port}")

    def stop(self) -> None:
        """Stop the Ray Serve deployment and shut down Ray."""
        print("[serving] shutting down Ray Serve")
        serve.shutdown()
        ray.shutdown()
