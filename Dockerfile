# Serving image: loads the latest registered model from the mounted registry
# and exposes the guarded prediction API on port 8000.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer first so code edits don't invalidate the resolve cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY configs ./configs
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Mount a trained registry at /app/artifacts/registry (see `make docker-serve`).
CMD ["mlops-serve", "--config", "configs/pipeline.yaml"]
