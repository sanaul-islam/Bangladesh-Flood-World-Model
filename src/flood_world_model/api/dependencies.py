from __future__ import annotations

from fastapi import Request

from flood_world_model.api.config import (
    DATABASE_PATH,
    POPULATION_RISK_PATH,
)

from flood_world_model.api.services import (
    FloodWorldModelService,
)


def create_service() -> FloodWorldModelService:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Road database not found: {DATABASE_PATH}"
        )

    if not POPULATION_RISK_PATH.exists():
        raise FileNotFoundError(
            "Population-risk dataset not found: "
            f"{POPULATION_RISK_PATH}"
        )

    return FloodWorldModelService(
        database_path=DATABASE_PATH,
        population_risk_path=POPULATION_RISK_PATH,
    )


def get_service(
    request: Request,
) -> FloodWorldModelService:

    service = getattr(
        request.app.state,
        "flood_service",
        None,
    )

    if service is None:
        raise RuntimeError(
            "Flood World Model service is not initialized."
        )

    if not service.ready:
        raise RuntimeError(
            "Flood World Model service is unavailable."
        )

    return service


def get_planning_limiter(
    request: Request,
):
    limiter = getattr(
        request.app.state,
        "planning_limiter",
        None,
    )

    if limiter is None:
        raise RuntimeError(
            "Planning limiter is not initialized."
        )

    return limiter
