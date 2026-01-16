# Base Stage
FROM python:3.12-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build Stage
FROM base AS builder

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv pip install --no-deps .

# Final Stage
FROM base AS final

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENTRYPOINT ["/app/.venv/bin/modelbench"]
