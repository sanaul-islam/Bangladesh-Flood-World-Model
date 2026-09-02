from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from flood_world_model.api.config import (
    ALLOWED_ORIGINS,
    MAX_CONCURRENT_PLANNING_REQUESTS,
)

from flood_world_model.api.concurrency import (
    PlanningLimiter,
)

from flood_world_model.api.dependencies import (
    create_service,
    get_planning_limiter,
    get_service,
)

from flood_world_model.api.observability import (
    EVACUATION_COUNT,
    HAZARD_COUNT,
    ROUTE_COUNT,
    configure_logging,
    metrics_response,
    request_metrics_middleware,
)

from flood_world_model.api.schemas import (
    EvacuationRequest,
    ForecastResponse,
    HazardResponse,
    HealthResponse,
    RouteRequest,
    SheltersResponse,
)

from flood_world_model.api.services import (
    FloodWorldModelService,
)

from flood_world_model.planning.routing import (
    RouteNotFoundError,
)


configure_logging()

logger = logging.getLogger(
    "flood_world_model.api"
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    service = None

    try:

        logger.info(
            "Starting Flood World Model API"
        )

        service = create_service()

        app.state.flood_service = service

        app.state.planning_limiter = (
            PlanningLimiter(
                MAX_CONCURRENT_PLANNING_REQUESTS
            )
        )

        logger.info(
            "Flood World Model service initialized"
        )

        yield

    except Exception:

        logger.exception(
            "API startup failed"
        )

        raise

    finally:

        if service is not None:

            try:

                service.close()

                logger.info(
                    "Flood World Model service closed"
                )

            except Exception:

                logger.exception(
                    "API shutdown failed"
                )

        app.state.flood_service = None
        app.state.planning_limiter = None


app = FastAPI(
    title="Bangladesh Flood World Model API",
    description=(
        "Forecast-aware flood risk and "
        "risk-aware evacuation decision "
        "support for Bangladesh."
    ),
    version="0.1.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Content-Type",
        "X-Request-ID",
    ],
)

app.middleware(
    "http"
)(
    request_metrics_middleware
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health(
    service: FloodWorldModelService = Depends(
        get_service
    ),
):
    status = service.health()

    return {
        "status": status[
            "status"
        ],
        "service": (
            "bangladesh-flood-world-model"
        ),
        "version": "0.1.1",
    }


@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics():
    return metrics_response()


@app.get(
    "/api/v1/forecast",
    response_model=ForecastResponse,
    tags=["forecast"],
)
def forecast(
    service: FloodWorldModelService = Depends(
        get_service
    ),
):

    try:

        return service.forecast_metadata()

    except Exception as error:

        logger.exception(
            "Forecast endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read "
                "forecast metadata."
            ),
        ) from error


@app.get(
    "/api/v1/hazard",
    response_model=HazardResponse,
    tags=["hazard"],
)
def hazard(
    latitude: float = Query(
        ...,
        ge=-90.0,
        le=90.0,
    ),
    longitude: float = Query(
        ...,
        ge=-180.0,
        le=180.0,
    ),
    forecast_sample: int = Query(
        0,
        ge=0,
    ),
    forecast_day: int = Query(
        1,
        ge=1,
        le=7,
    ),
    service: FloodWorldModelService = Depends(
        get_service
    ),
):

    try:

        result = service.hazard(
            latitude=latitude,
            longitude=longitude,
            forecast_sample=forecast_sample,
            forecast_day=forecast_day,
        )

        HAZARD_COUNT.labels(
            status="success"
        ).inc()

        return result

    except ValueError as error:

        HAZARD_COUNT.labels(
            status="client_error"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:

        HAZARD_COUNT.labels(
            status="processing_error"
        ).inc()

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:

        HAZARD_COUNT.labels(
            status="server_error"
        ).inc()

        logger.exception(
            "Hazard endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate hazard."
            ),
        ) from error
@app.get(
    "/api/v1/hazard/grid",
    tags=["hazard"],
)
def hazard_grid(
    forecast_sample: int = Query(
        0,
        ge=0,
    ),
    forecast_day: int = Query(
        1,
        ge=1,
        le=7,
    ),
    center_latitude: float = Query(
        23.8103,
        ge=-90.0,
        le=90.0,
    ),
    center_longitude: float = Query(
        90.4125,
        ge=-180.0,
        le=180.0,
    ),
    radius_degrees: float = Query(
        0.30,
        gt=0.01,
        le=2.0,
    ),
    rows: int = Query(
        9,
        ge=3,
        le=15,
    ),
    columns: int = Query(
        9,
        ge=3,
        le=15,
    ),
    service: FloodWorldModelService = Depends(
        get_service
    ),
):
    try:
        lat_min = center_latitude - radius_degrees
        lat_max = center_latitude + radius_degrees

        lon_min = center_longitude - radius_degrees
        lon_max = center_longitude + radius_degrees

        lat_step = (
            (lat_max - lat_min) / (rows - 1)
            if rows > 1
            else 0.0
        )

        lon_step = (
            (lon_max - lon_min) / (columns - 1)
            if columns > 1
            else 0.0
        )

        points = []

        for row in range(rows):
            latitude = lat_min + row * lat_step

            for column in range(columns):
                longitude = (
                    lon_min + column * lon_step
                )

                result = service.hazard(
                    latitude=latitude,
                    longitude=longitude,
                    forecast_sample=forecast_sample,
                    forecast_day=forecast_day,
                )

                points.append(
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "hazard_score": float(
                            result["hazard_score"]
                        ),
                    }
                )

        return {
            "forecast_sample": forecast_sample,
            "forecast_day": forecast_day,
            "center_latitude": center_latitude,
            "center_longitude": center_longitude,
            "radius_degrees": radius_degrees,
            "rows": rows,
            "columns": columns,
            "points": points,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

@app.get(
    "/api/v1/shelters",
    response_model=SheltersResponse,
    tags=["shelters"],
)
def shelters(
    service: FloodWorldModelService = Depends(
        get_service
    ),
):

    try:

        return service.shelters()

    except Exception as error:

        logger.exception(
            "Shelters endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read shelters."
            ),
        ) from error


@app.post(
    "/api/v1/route",
    tags=["routing"],
)
async def route(
    request: RouteRequest,
    service: FloodWorldModelService = Depends(
        get_service
    ),
    limiter: PlanningLimiter = Depends(
        get_planning_limiter
    ),
):

    try:

        async with limiter:

            result = service.route(
                start_latitude=(
                    request.start_latitude
                ),
                start_longitude=(
                    request.start_longitude
                ),
                goal_latitude=(
                    request.goal_latitude
                ),
                goal_longitude=(
                    request.goal_longitude
                ),
                max_distance_km=(
                    request.max_distance_km
                ),
                max_expanded_nodes=(
                    request.max_expanded_nodes
                ),
            )

        ROUTE_COUNT.labels(
            status="success"
        ).inc()

        return {
            "status": "ok",
            "result": result,
        }

    except RouteNotFoundError as error:

        ROUTE_COUNT.labels(
            status="not_found"
        ).inc()

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:

        ROUTE_COUNT.labels(
            status="client_error"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        ROUTE_COUNT.labels(
            status="server_error"
        ).inc()

        logger.exception(
            "Route endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate route."
            ),
        ) from error


@app.post(
    "/api/v1/evacuate",
    tags=["evacuation"],
)
async def evacuate(
    request: EvacuationRequest,
    service: FloodWorldModelService = Depends(
        get_service
    ),
    limiter: PlanningLimiter = Depends(
        get_planning_limiter
    ),
):

    try:

        async with limiter:

            result = service.evacuate(
                latitude=request.latitude,
                longitude=request.longitude,
                forecast_sample=(
                    request.forecast_sample
                ),
                forecast_day=(
                    request.forecast_day
                ),
                candidate_shelters=(
                    request.candidate_shelters
                ),
            )

        EVACUATION_COUNT.labels(
            status="success"
        ).inc()

        return {
            "status": "ok",
            "result": result,
        }

    except RouteNotFoundError as error:

        EVACUATION_COUNT.labels(
            status="not_found"
        ).inc()

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:

        EVACUATION_COUNT.labels(
            status="client_error"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:

        EVACUATION_COUNT.labels(
            status="processing_error"
        ).inc()

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:

        EVACUATION_COUNT.labels(
            status="server_error"
        ).inc()

        logger.exception(
            "Evacuation endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate "
                "evacuation recommendation."
            ),
        ) from error
