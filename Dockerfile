FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Keep uv from writing a large persistent cache.
ENV UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /app

# ---------------------------------------------------------
# Only runtime tools required by the container healthcheck.
# Python wheels provide the required native Python libraries.
# ---------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Non-root application user
# ---------------------------------------------------------
RUN useradd \
    --create-home \
    --uid 10001 \
    --shell /usr/sbin/nologin \
    floodapi

# ---------------------------------------------------------
# Install uv
# ---------------------------------------------------------
RUN pip install --no-cache-dir uv

# ---------------------------------------------------------
# Dependency metadata
# ---------------------------------------------------------
COPY --chown=floodapi:floodapi \
    pyproject.toml \
    uv.lock \
    README.md \
    ./

# ---------------------------------------------------------
# Install production dependencies only.
# NO research extra.
# NO dev dependencies.
# ---------------------------------------------------------
RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project \
    --no-editable \
    --no-cache

# ---------------------------------------------------------
# Application source
# ---------------------------------------------------------
COPY --chown=floodapi:floodapi src ./src

# ---------------------------------------------------------
# Install application
# ---------------------------------------------------------
RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable \
    --no-cache

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

CMD ["uv", "run", "--no-dev", "--no-sync", "uvicorn", "flood_world_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
