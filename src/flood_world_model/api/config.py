from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


def env_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer."
        ) from error


def env_float(
    name: str,
    default: float,
) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a number."
        ) from error


def env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


DATABASE_PATH = Path(
    os.getenv(
        "FLOOD_DATABASE_PATH",
        str(
            PROJECT_ROOT
            / "data/processed/road_network.sqlite"
        ),
    )
)

POPULATION_RISK_PATH = Path(
    os.getenv(
        "FLOOD_POPULATION_RISK_PATH",
        str(
            PROJECT_ROOT
            / "outputs/predictions/"
            "v2_population_population_risk.nc"
        ),
    )
)

API_HOST = os.getenv(
    "FLOOD_API_HOST",
    "0.0.0.0",
)

API_PORT = env_int(
    "FLOOD_API_PORT",
    8000,
)

API_LOG_LEVEL = os.getenv(
    "FLOOD_API_LOG_LEVEL",
    "INFO",
)

_allowed_origins = os.getenv(
    "FLOOD_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _allowed_origins.split(",")
    if origin.strip()
]

MAX_CANDIDATE_SHELTERS = env_int(
    "FLOOD_MAX_CANDIDATE_SHELTERS",
    24,
)

MAX_ROUTE_DISTANCE_KM = env_float(
    "FLOOD_MAX_ROUTE_DISTANCE_KM",
    30.0,
)

MAX_ROUTE_EXPANDED_NODES = env_int(
    "FLOOD_MAX_ROUTE_EXPANDED_NODES",
    100000,
)

MAX_CONCURRENT_PLANNING_REQUESTS = max(
    1,
    env_int(
        "FLOOD_MAX_CONCURRENT_PLANNING_REQUESTS",
        1,
    ),
)
