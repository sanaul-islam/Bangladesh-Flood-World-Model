# Bangladesh Flood World Model

## Forecast-Aware Flood Risk & Risk-Aware Evacuation Platform

A geospatial AI and disaster-response decision-support system for Bangladesh that connects **spatiotemporal flood forecasting, forecast uncertainty, population exposure, infrastructure risk, risk-aware routing, and automatic evacuation-shelter recommendation** into a deployable API.

> **Forecast → Hazard → Exposure → Infrastructure Risk → Routing → Evacuation**

Built with PyTorch, Xarray, GeoPandas, OSM/OSMnx, SQLite, FastAPI, Prometheus, Docker, GitHub Actions, GitHub Container Registry, and AWS EC2.

---

# Why This Project?

Flood prediction by itself is not an evacuation system.

A useful response system must answer a more operational question:

> **Given a person's current location and the predicted flood state, which evacuation shelter can they reach most safely and efficiently?**

This project therefore combines environmental forecasting with infrastructure-aware decision making.

The system takes a forecasted hydrological state and propagates it into:

1. Flood hazard
2. Population exposure
3. Road-level risk
4. Bridge exposure
5. Risk-aware routing
6. Shelter ranking
7. Automatic evacuation recommendation

The result is an end-to-end research and engineering pipeline rather than an isolated prediction model.

---

# System Overview

```text
                         DATA SOURCES

                              │
             ┌────────────────┼────────────────┐
             │                │                │
         Rainfall         Discharge       Static GIS
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                 V2-Population World Model
                              │
                       60 × 45 grid
                              │
                         7-day rollout
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       Forecast Uncertainty          Hydrological State
                │                           │
                └─────────────┬─────────────┘
                              ▼
                     Population Exposure
                              │
                              ▼
                Infrastructure Risk Layer
                              │
              ┌───────────────┼───────────────┐
              │               │               │
            Roads           Bridges       Uncertainty
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                  SQLite Road Network
                              │
                              ▼
                       Risk-Aware A*
                              │
                              ▼
                     Shelter Candidates
                              │
                              ▼
                       Shelter Ranking
                              │
                              ▼
                   Evacuation Recommendation
                              │
                              ▼
                         FastAPI API
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
             Client        Monitoring     Docker
```

---

# Key Results & System Scale

## Forecasting

The world model currently operates over:

| Property         |           Value |
| ---------------- | --------------: |
| Spatial grid     |     **60 × 45** |
| Forecast horizon |      **7 days** |
| Forecast samples |         **606** |
| Latitude range   | ~20.55°–26.45°N |
| Longitude range  | ~88.05°–92.45°E |

### Current verified V2-Population evaluation

| Metric                    |            Result |
| ------------------------- | ----------------: |
| Pooled 7-day RMSE         |  **672.533 m³/s** |
| Persistence baseline RMSE | **~993.996 m³/s** |
| RMSE skill vs persistence |        **32.34%** |

The model is evaluated against a persistence baseline using the current held-out evaluation protocol.

### High-flow event evaluation

The current canonical event analysis reports:

| Metric                       |         Result |
| ---------------------------- | -------------: |
| Events analyzed              |         **76** |
| Mean relative peak error     |     **~0.01%** |
| Mean peak timing error       | **~0.45 days** |
| Severe underprediction cases |          **0** |

These values represent the current validated evaluation configuration.

---

# Forecast Uncertainty

The system includes an explicit uncertainty layer based on forecast residual behavior.

Instead of producing only:

```text
prediction
```

the system maintains:

```text
prediction + uncertainty
```

for downstream risk propagation.

Current evaluated interval configuration:

```text
Target empirical coverage:   80%
Observed coverage:           80%
```

The uncertainty layer is then incorporated into road and evacuation risk rather than discarded after forecasting.

---

# Hydrological Hazard

Forecast variables are transformed into a normalized hazard representation for downstream planning.

The current hazard pipeline incorporates information derived from:

* hydrological/discharge state
* rainfall
* terrain/elevation
* slope
* river-network context
* forecast uncertainty

The resulting forecast artifact is reused by the planning layer rather than recomputed on every API request.

---

# Population Exposure

Population information is integrated with the forecast hazard field.

The population-risk artifact contains:

```text
hydrological_hazard_score
population_component
population_exposure_index
population_density
```

The resulting exposure field is used both for destination assessment and evacuation-shelter ranking.

### Shelter forecast validation

Current validation:

```text
Valid shelters:     281 / 281
Invalid shelters:     0
```

This validation prevents missing forecast values from silently becoming misleading risk values.

---

# Large-Scale Geospatial Infrastructure

A nationwide in-memory NetworkX graph produced significant memory pressure during development.

Instead, the project uses a **disk-backed SQLite road network**.

Current network scale:

| Resource             |         Count |
| -------------------- | ------------: |
| Road nodes           | **4,281,503** |
| Road edges           | **4,304,762** |
| Bridges              |    **25,500** |
| Bridge-node mappings |    **19,861** |
| Bridge-edge mappings |    **37,490** |
| Shelters             |       **281** |
| Road-mapped shelters |       **274** |

This is an intentional systems-engineering decision:

```text
Large geospatial network
        ↓
Indexed SQLite storage
        ↓
Local/network queries
        ↓
A*
```

rather than:

```text
4.3M-edge nationwide graph
        ↓
load everything into RAM
```

The result is substantially more memory-aware for CPU-only environments.

---

# Road Risk Engine

The road network is enriched with forecast-aware risk.

Current validated coverage:

```text
Road edges:                  4,304,762
Road-risk records:           4,304,762
Nonzero flood-risk records:  4,304,762
```

Each edge can carry information relating to:

* flood risk
* forecast uncertainty
* bridge exposure
* travel-time cost
* total route cost

Current conceptual risk formulation:

```text
risk_cost =

    travel_time

    ×

    (
        1
        + 2 × flood_risk
        + 2 × bridge_risk
        + 1 × uncertainty_risk
    )
```

This converts the forecast into an infrastructure-aware cost surface for route planning.

---

# Bridge-Aware Planning

Bridges are explicitly integrated into the road-risk layer.

Current infrastructure:

```text
Bridge features:          25,500
Bridge-node mappings:     19,861
Bridge-edge mappings:     37,490
```

Routes expose bridge-associated edges so evacuation recommendations can distinguish between:

```text
short route
```

and:

```text
shorter/safer route with lower bridge exposure
```

---

# Risk-Aware Routing

The project implements a **SQLite-backed A*** routing engine.

The router:

* snaps arbitrary geographic coordinates to nearby road nodes
* searches the relevant road network
* uses risk-aware edge costs
* reconstructs the route
* converts network nodes into geographic coordinates
* returns route statistics

### Example validated route

```text
Route nodes:              226
Road distance:            9.86 km
Estimated travel time:    12.29 min
```

The route response includes:

```text
nodes
edge_ids
coordinates
road_distance_km
estimated_travel_time_min
risk_cost
mean_flood_risk
maximum_flood_risk
mean_uncertainty_risk
maximum_uncertainty_risk
maximum_bridge_risk
bridge_edges
```

---

# Automatic Evacuation Recommendation

The evacuation planner evaluates multiple candidate shelters and ranks them using destination and route-level risk.

Current ranking factors:

```text
35%  Route risk
20%  Destination hazard
20%  Population exposure
15%  Travel time
10%  Bridge exposure
```

The system therefore evaluates both:

```text
"How dangerous is the destination?"
```

and:

```text
"How dangerous is the journey to reach it?"
```

### Example

For a test user location:

```text
Latitude:       23.8103
Longitude:      90.4125
```

the current validated system produced:

```text
Recommended shelter:       256
Road distance:             ~1.79 km
Estimated travel time:     ~2.91 min
Bridge-associated edges:   2
```

The API also returns alternative shelters and their ranking statistics.

---

# Decision Output

A recommendation contains more than a shelter ID.

The system returns:

```text
User location
Nearest road node
Shelter location
Road distance
Estimated travel time
Route geometry
Flood risk
Uncertainty risk
Bridge exposure
Destination hazard
Population exposure
Population density
Ranking score
Alternative shelters
Forecast sample
Forecast day
```

This makes the output auditable rather than producing an unexplained classification.

---

# FastAPI Backend

The planning system is exposed through a versioned REST API.

## Endpoints

### Health

```http
GET /health
```

Returns service health and verifies that the backend has initialized correctly.

### Forecast

```http
GET /api/v1/forecast
```

Returns forecast metadata including:

* available forecast samples
* forecast days
* spatial dimensions
* geographic extent

### Hazard

```http
GET /api/v1/hazard
```

Example:

```text
/api/v1/hazard?latitude=23.8103&longitude=90.4125&forecast_sample=0&forecast_day=1
```

Returns the forecast-aware hazard and population exposure for a requested coordinate.

### Shelters

```http
GET /api/v1/shelters
```

Returns the shelter inventory and road-network mapping state.

### Route

```http
POST /api/v1/route
```

Calculates a risk-aware route between two coordinates.

Example:

```json
{
  "start_latitude": 23.8103,
  "start_longitude": 90.4125,
  "goal_latitude": 23.80208345,
  "goal_longitude": 90.40965735
}
```

### Evacuation

```http
POST /api/v1/evacuate
```

Example:

```json
{
  "latitude": 23.8103,
  "longitude": 90.4125,
  "forecast_sample": 0,
  "forecast_day": 1,
  "candidate_shelters": 5
}
```

The endpoint executes the complete evacuation-ranking pipeline.

### Metrics

```http
GET /metrics
```

Exposes Prometheus-compatible metrics.

---

# API Documentation

When the service is running locally:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

The same API is deployed on AWS EC2 for cloud validation.

---

# Production-Oriented Backend Design

The backend is designed around reusable initialized resources rather than recreating expensive objects for each request.

```text
FastAPI startup

      ↓

Initialize forecast artifact

Initialize SQLite-backed planning service

      ↓

Reuse service across requests

      ↓

FastAPI shutdown

      ↓

Close resources cleanly
```

The service also protects expensive routing/planning operations with bounded concurrency.

The scientific road database is consumed as a **read-only artifact** by the serving layer.

---

# Observability

The API contains application-level observability for production-style operation.

## Structured logs

Logs are emitted as structured JSON records.

Example fields include:

```text
timestamp
level
logger
message
request_id
```

## Request IDs

Each HTTP request receives an `X-Request-ID`.

Client-supplied request IDs can also be propagated.

## Prometheus metrics

The service tracks:

```text
flood_api_requests_total
flood_api_request_duration_seconds
flood_api_evacuations_total
flood_api_routes_total
flood_api_hazard_queries_total
```

This allows the model-serving layer to be monitored independently from offline model training.

---

# Docker

The backend is containerized using Docker and Docker Compose.

The production container:

* runs as a non-root user
* mounts large artifacts at runtime
* treats scientific artifacts as read-only
* uses environment-based configuration
* performs a healthcheck
* limits expensive planning concurrency
* drops Linux capabilities
* enables `no-new-privileges`
* exposes Prometheus metrics

The large national road database and forecast artifacts are intentionally **not baked into the Docker image**.

## Production image optimization

The original production image contained the full research/training environment and reached approximately:

```text
18.8 GB
```

The production dependency tree was separated from the research/training dependency tree, removing unnecessary packages such as the training PyTorch stack from the serving image.

The resulting production image is approximately:

```text
~693 MB
```

This reduces deployment size by roughly **96%** while retaining the dependencies required by the API and routing stack.

---

# CI / Testing

The project uses automated testing for both the application and planning pipeline.

## Unit tests

```bash
uv run pytest tests/unit -v
```

## API integration tests

```bash
uv run pytest tests/integration/test_api.py -v
```

## Evacuation pipeline integration test

```bash
uv run pytest tests/integration/test_evacuation_pipeline.py -v
```

## Docker smoke tests

```bash
RUN_DOCKER_TESTS=1 uv run pytest \
  tests/integration/test_docker_api.py -v
```

The integration tests validate the complete chain from API request through routing and shelter recommendation.

---

# CI/CD

The project implements automated CI/CD using **GitHub Actions, GitHub Container Registry, Docker, and AWS EC2**.

## Continuous Integration

The CI pipeline validates changes before deployment.

```text
git push
   ↓
GitHub Actions
   ↓
Dependency installation
   ↓
Automated tests
   ↓
Docker validation
```

## Container Registry

Production container images are published to:

```text
ghcr.io/sanaul-islam/bangladesh-flood-world-model
```

Current deployed container release:

```text
v0.1.1
```

## Continuous Deployment

A self-hosted GitHub Actions runner operates directly on the AWS EC2 instance.

```text
git push
   ↓
GitHub Actions
   ↓
CD job
   ↓
Self-hosted runner on AWS EC2
   ↓
docker compose pull
   ↓
docker compose up -d
   ↓
health check
   ↓
deployment complete
```

The deployment process automatically verifies the application health after updating the service.

The EC2 runner is installed as a persistent service so it can resume after a machine restart.

---

# AWS Cloud Deployment

The production API has been deployed and validated on **Amazon EC2**.

## Current deployment

| Component         | Configuration              |
| ----------------- | -------------------------- |
| Cloud             | **AWS EC2**                |
| Region            | **eu-north-1 (Stockholm)** |
| Instance          | **t3.small**               |
| Architecture      | **x86_64**                 |
| Memory            | **2 GiB**                  |
| Root storage      | **20 GiB**                 |
| Operating system  | **Ubuntu 26.04 LTS**       |
| Container runtime | **Docker**                 |
| Container image   | **GHCR `v0.1.1`**          |
| Image size        | **~693 MB**                |

## Runtime artifacts

Large scientific artifacts are stored separately from the Docker image and mounted into the production container:

```text
/data/flood-world-model/
├── processed/
│   └── road_network.sqlite
└── predictions/
    └── v2_population_population_risk.nc
```

The serving container consumes these artifacts as read-only inputs.

## Cloud validation

The deployed API has been validated using the actual production artifacts.

Verified successfully:

```text
FastAPI startup
       ↓
SQLite road database
       ↓
Forecast-risk NetCDF
       ↓
Risk-aware routing
       ↓
Shelter ranking
       ↓
Evacuation recommendation
```

A complete evacuation request was successfully executed on the 2 GiB EC2 instance.

Observed resource usage during the validated request:

```text
Container memory:       ~493 MiB
EC2 available RAM:      ~959 MiB
Swap configured:        2 GiB
Swap used during test:  negligible
```

This demonstrates that the current CPU-only serving architecture can operate on a resource-constrained cloud instance.

---

# Production Security & Runtime Controls

The production container includes several hardening measures:

```text
Non-root container user
Read-only root filesystem
Read-only scientific artifacts
No-new-privileges
Linux capability dropping
Temporary writable /tmp
Bounded planning concurrency
Container healthcheck
Automatic container restart
```

The SQLite and forecast artifacts are intentionally separated from the container image.

The current deployment keeps FastAPI on the application port internally; a future reverse-proxy/HTTPS layer is planned for public production access.

---

# Repository Structure

```text
Bangladesh-Flood-World-Model/

│
├── src/
│   └── flood_world_model/
│       ├── api/
│       ├── data/
│       ├── datasets/
│       ├── evaluation/
│       ├── hazard/
│       ├── inference/
│       ├── models/
│       ├── planning/
│       ├── risk/
│       ├── training/
│       ├── utils/
│       └── visualization/
│
├── scripts/
│   ├── inference/
│   ├── planning/
│   ├── risk/
│   ├── hazard/
│   ├── train_v2.py
│   ├── train_v2_population.py
│   ├── evaluate_v2_7day.py
│   ├── evaluate_v2_population_events.py
│   ├── evaluate_v2_vs_population.py
│   └── ...
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── configs/
│
├── outputs/
│   └── metrics/
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

Large generated datasets and runtime artifacts are intentionally excluded from Git.

---

# Technology Stack

## Machine Learning

* Python 3.11+
* PyTorch
* ConvGRU / spatiotemporal sequence modeling
* NumPy
* Xarray
* Dask

## Geospatial

* GeoPandas
* Shapely
* PyProj
* PyOgrio
* OSMnx
* Osmium
* Cartopy
* Rioxarray

## Storage

* SQLite
* NetCDF / HDF5
* Zarr
* Pandas

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Observability

* Prometheus Client
* Structured JSON logging

## Testing

* pytest
* FastAPI TestClient
* Docker smoke tests

## Tooling

* uv
* Hatchling

## Deployment

* Docker
* Docker Compose
* GitHub Actions
* GitHub Container Registry
* AWS EC2

---

# Quick Start

## Requirements

```text
Python 3.11+
uv
Docker
Docker Compose
```

## Install

```bash
uv sync --dev
```

Verify:

```bash
uv run python -c \
  "import flood_world_model; print('package OK')"
```

## Run tests

```bash
uv run pytest tests/unit -v
uv run pytest tests/integration/test_api.py -v
uv run pytest tests/integration/test_evacuation_pipeline.py -v
```

## Run the API

```bash
uv run uvicorn flood_world_model.api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

```bash
cp .env.example .env

docker compose build

docker compose up
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

---

# Production Deployment

The production API uses:

```text
docker-compose.prod.yml
```

The production deployment expects the large runtime artifacts to be present outside the Docker image.

Example:

```text
/data/flood-world-model/processed/road_network.sqlite
/data/flood-world-model/predictions/v2_population_population_risk.nc
```

The production image can be pulled from GHCR:

```bash
docker pull ghcr.io/sanaul-islam/bangladesh-flood-world-model:v0.1.1
```

The current AWS deployment uses:

```bash
docker compose -f docker-compose.prod.yml up -d
```

---

# Data & Artifact Strategy

The project deliberately separates **code** from **large scientific artifacts**.

## Stored in Git

* application source code
* training/evaluation scripts
* tests
* configuration
* documentation
* Docker configuration
* CI/CD configuration
* small evaluation metrics

## Stored outside normal Git history

Examples:

```text
data/raw/
data/static/
data/interim/
data/features/
data/processed/

outputs/predictions/

model checkpoints

large raster/vector datasets
```

The national road SQLite database is approximately **1.6 GB** and is therefore not included in the Git repository.

The same principle applies to large NetCDF, Zarr, GRIB, PBF, TIFF, and other generated/source artifacts.

Production serving artifacts are mounted separately from the container image.

---

# Engineering Principles

## 1. Model and serving separation

Expensive operations happen offline:

```text
Data processing
      ↓
Model training
      ↓
Forecast generation
      ↓
Risk artifact generation
      ↓
Validated artifacts
      ↓
FastAPI serving
```

API requests do not retrain the model or rebuild the national road network.

## 2. Memory-aware geospatial engineering

The project avoids constructing the complete nationwide road graph in RAM.

Instead:

```text
SQLite
  ↓
Indexed network data
  ↓
Local database queries
  ↓
A*
```

## 3. Explicit uncertainty

Forecast uncertainty is retained and propagated into downstream infrastructure risk.

## 4. Read-only serving artifacts

Validated forecasting and road-network artifacts are treated as immutable serving inputs.

## 5. Explainable decision output

The evacuation endpoint returns the factors behind the recommendation rather than only returning a destination identifier.

## 6. Resource-aware deployment

The serving architecture is designed to operate without a GPU and has been validated on a 2 GiB AWS EC2 instance.

---

# Current Limitations

This project is a research and engineering prototype and **not an operational emergency-response system**.

Current limitations include:

* approximately 0.1° forecast spatial resolution
* no validated live shelter occupancy/capacity feed
* no live road-closure feed
* forecast artifacts are not yet continuously refreshed in production
* route risk depends on the currently generated forecast artifact
* candidate-shelter evaluation is configurable rather than exhaustive for every request
* no production HTTPS/domain layer yet
* current application version is `0.1.0` while the deployed container image is `v0.1.1`
* operational deployment requires authoritative, real-time government and emergency-management data

These limitations are explicitly represented in the system instead of being hidden.

---

# Roadmap

## Completed

* [x] Spatiotemporal flood world model
* [x] V2-Population forecasting
* [x] 7-day autoregressive forecasting
* [x] Forecast uncertainty
* [x] Held-out evaluation
* [x] High-flow event evaluation
* [x] Hydrological hazard
* [x] Population exposure
* [x] Nationwide OSM road network
* [x] Bridge integration
* [x] Road-level risk
* [x] Uncertainty-aware routing
* [x] SQLite-backed A*
* [x] Shelter database
* [x] Shelter-road mapping
* [x] Shelter forecast validation
* [x] Automatic shelter ranking
* [x] FastAPI backend
* [x] Structured logging
* [x] Request IDs
* [x] Prometheus metrics
* [x] Docker
* [x] Docker healthchecks
* [x] Production dependency separation
* [x] Production Docker image optimization
* [x] Integration testing
* [x] GitHub Actions CI
* [x] GitHub Container Registry publishing
* [x] AWS EC2 deployment
* [x] Self-hosted GitHub Actions runner
* [x] Automated EC2 deployment
* [x] Cloud validation on a 2 GiB instance

## Next

* [ ] Version-aligned API/Container releases
* [ ] Immutable image deployment by commit SHA
* [ ] Automated forecast/artifact refresh
* [ ] Live FFWC data integration
* [ ] Live road-status integration
* [ ] Live shelter availability
* [ ] Interactive map dashboard
* [ ] HTTPS/domain deployment
* [ ] API authentication and rate limiting
* [ ] Production monitoring dashboards
* [ ] Model/data drift monitoring
* [ ] Automated data/model versioning
* [ ] Multi-region/cloud deployment

---

# Responsible Use

This system is intended for:

* research
* engineering evaluation
* disaster-response prototyping
* decision-support experimentation

It should **not be used as the sole source of emergency instructions**.

Operational decisions should incorporate:

* official government warnings
* current flood observations
* verified road conditions
* confirmed shelter availability
* emergency-management coordination
* human judgment

---

# Data and Third-Party Components

The source code in this repository is licensed under the MIT License.

Third-party datasets, map data, pretrained models, libraries, APIs, and external services remain subject to their respective licenses, terms of use, and attribution requirements.

Users are responsible for complying with the applicable licenses and attribution requirements of external resources.

---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

# Author

**Md Sanaul Islam**

Computer Science & Engineering
BRAC University, Bangladesh
