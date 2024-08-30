# Base Stage
FROM python:3.10-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y \
    git
WORKDIR /app

# Build Stage
FROM base AS builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Copy Poetry files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN . /venv/bin/activate && poetry install --without=dev --no-root --no-interaction --no-ansi

COPY . .
RUN . /venv/bin/activate && poetry build

# Final Stage
FROM base AS final

WORKDIR /app


# Pass the GITHUB_TOKEN to the final stage
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Copy the virtual environment and built package
COPY --from=builder /venv /venv
COPY --from=builder /app/dist .

RUN . /venv/bin/activate && pip install *.whl
ENTRYPOINT ["/venv/bin/modelbench", "--help"]