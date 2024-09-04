# Base Stage
FROM python:3.10-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build Stage
FROM base AS builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN . /venv/bin/activate && poetry install --without=dev --no-root --no-interaction --no-ansi

COPY . .
RUN . /venv/bin/activate && poetry build

# Final Stage
FROM base AS final

ARG PIP_EXTRA=false

WORKDIR /app

COPY --from=builder /venv /venv
COPY --from=builder /app/dist .

RUN . /venv/bin/activate \
    && pip install *.whl \
    && if [ "$PIP_EXTRA" != "false" ] ; then pip install "$PIP_EXTRA"; fi
ENTRYPOINT ["/venv/bin/modelbench", "--help"]