FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        proj-bin \
        proj-data \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./

RUN pip install --no-cache-dir uv

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

COPY src ./src

RUN uv sync \
    --frozen \
    --no-dev

RUN useradd \
    --create-home \
    --uid 10001 \
    --shell /usr/sbin/nologin \
    floodapi

RUN mkdir -p \
    /app/data \
    /app/outputs \
    && chown -R floodapi:floodapi /app

USER floodapi

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl \
        --fail \
        --silent \
        http://127.0.0.1:8000/health \
        || exit 1

CMD ["uv", "run", "--no-dev", "uvicorn", "flood_world_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
