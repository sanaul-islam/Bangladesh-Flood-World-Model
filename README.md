# Bangladesh Flood World Model

Forecast-aware flood risk and risk-aware evacuation decision support for Bangladesh.

## Core Components

- Spatiotemporal V2-Population flood world model
- 7-day autoregressive forecasting
- Forecast uncertainty estimation
- Hydrological hazard modeling
- Population exposure modeling
- Disk-backed OSM road network
- Bridge-aware road risk
- SQLite-backed risk-aware A* routing
- 281 shelter locations
- Automatic shelter ranking
- FastAPI backend
- Prometheus metrics and structured logging

## Backend

The API exposes:

- `GET /health`
- `GET /api/v1/forecast`
- `GET /api/v1/hazard`
- `GET /api/v1/shelters`
- `POST /api/v1/route`
- `POST /api/v1/evacuate`

API documentation is available at `/docs` when the server is running.

## Run


uv sync
uv run uvicorn flood_world_model.api.app:app --host 127.0.0.1 --port 8000


## Docker

docker compose build
docker compose up
