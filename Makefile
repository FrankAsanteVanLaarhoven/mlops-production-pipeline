.PHONY: install lint format test test-all train serve smoke docker-build docker-serve clean

install:            ## Install all dependencies including dev tools
	uv sync --group dev

lint:               ## Static checks
	uv run ruff check src tests

format:             ## Auto-format
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:               ## Unit tests (fast, no orchestration stack)
	uv run pytest -m "not integration" -q

test-all:           ## Everything, including the end-to-end pipeline test
	uv run pytest -q

train:              ## Run the full training pipeline
	uv run mlops-train --config configs/pipeline.yaml

serve:              ## Serve the latest registered model
	uv run mlops-serve --config configs/pipeline.yaml

smoke:              ## Deploy, hit the guarded API with good/bad requests, exit
	uv run mlops-serve --config configs/pipeline.yaml --smoke-test

docker-build:       ## Build the serving image
	docker build -t mlops-production-pipeline:latest .

docker-serve:       ## Serve inside Docker (mounts the local registry read-only)
	docker run --rm -p 8000:8000 \
		-v $(PWD)/artifacts/registry:/app/artifacts/registry:ro \
		mlops-production-pipeline:latest

clean:              ## Remove caches and staging artifacts
	rm -rf .pytest_cache .ruff_cache artifacts/staging
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
