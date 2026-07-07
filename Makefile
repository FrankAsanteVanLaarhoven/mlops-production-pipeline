.PHONY: install hooks lint review test test-all format train serve smoke notebook docker-build docker-serve clean

install:            ## Install all dependencies including dev + notebook tools
	uv sync --group dev --group notebooks

hooks:              ## Install git hooks (review + lint on commit, tests on push)
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

lint:               ## Static checks
	uv run ruff check src tools tests

review:             ## Function-level code review (docstrings, types, complexity, churn)
	uv run python tools/code_review.py

format:             ## Auto-format
	uv run ruff format src tools tests
	uv run ruff check --fix src tools tests

test:               ## Unit tests with coverage gate (fast, no orchestration stack)
	uv run pytest -m "not integration" -q --cov --cov-fail-under=95

test-all:           ## Everything, including the end-to-end pipeline test
	uv run pytest -q

notebook:           ## Execute the case-study notebook end-to-end
	uv run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 \
		notebooks/adult_income_case_study.ipynb

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
