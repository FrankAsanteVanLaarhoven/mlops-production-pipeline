"""Guarded model serving entry point."""

import argparse
import os
import sys
import time
from pathlib import Path

from .adapters.ray_serving import RayServingAdapter
from .config import PipelineConfig
from .core.serving import BaseServingService

# Registry of available serving services
_SERVING_SERVICES: dict[str, type[BaseServingService]] = {
    "ray": RayServingAdapter,
}


def register_serving_service(name: str, service_cls: type[BaseServingService]) -> None:
    """Register a custom serving engine (e.g. kserve, triton)."""
    _SERVING_SERVICES[name] = service_cls


def get_serving_service(name: str = "ray") -> BaseServingService:
    """Retrieve a serving service engine instance by name."""
    if name not in _SERVING_SERVICES:
        raise ValueError(
            f"Serving service '{name}' is not registered. "
            f"Choose from: {list(_SERVING_SERVICES.keys())}"
        )
    return _SERVING_SERVICES[name]()


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
    parser.add_argument(
        "--engine",
        default="ray",
        help="serving engine to use (default: ray)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    registry_root = (
        args.registry_root or os.environ.get("MODEL_REGISTRY_ROOT") or str(cfg.registry.root)
    )

    serving_service = get_serving_service(args.engine)
    serving_service.start(
        host=cfg.serving.host,
        port=cfg.serving.port,
        registry_root=registry_root,
        max_abs_feature_value=cfg.serving.max_abs_feature_value,
        production_data_path=str(cfg.serving.production_data_path),
    )

    if args.smoke_test:
        exit_code = _run_smoke_test(f"http://127.0.0.1:{cfg.serving.port}")
        serving_service.stop()
        sys.exit(exit_code)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        serving_service.stop()


if __name__ == "__main__":
    main()
