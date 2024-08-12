# Base Stage
FROM python:3.10-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y \
    gcc \
    libffi-dev \
    g++ \
    git

WORKDIR /app

# Final Stage
FROM base AS final

RUN ["pip", "install", "git+https://github.com/mlcommons/modelbench.git"]

ENTRYPOINT ["modelbench", "benchmark", "-m", "1"]