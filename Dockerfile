# Base Stage
FROM python:3.12-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl pipx \
    && rm -rf /var/lib/apt/lists/* \
    && pipx install uv

WORKDIR /app

# Build Stage
FROM base AS builder

ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT="/app/venv"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Final Stage
FROM base AS final

ENV PATH="/root/.local/bin:${PATH}"
ENV VIRTUAL_ENV="/app/venv"

WORKDIR /app

COPY --from=builder /app/venv /app/venv

COPY . .

RUN /app/venv/bin/pip install --no-deps -e .

ENTRYPOINT ["/app/venv/bin/modelbench"]
