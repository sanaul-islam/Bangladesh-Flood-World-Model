from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ForecastResponse(BaseModel):
    forecast_sample: int
    forecast_days: list[int]
    latitude_points: int
    longitude_points: int
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


class LocationRequest(BaseModel):
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
    )


class HazardRequest(LocationRequest):
    forecast_sample: int = Field(
        default=0,
        ge=0,
    )
    forecast_day: int = Field(
        default=1,
        ge=0,
        le=6,
    )


class RouteRequest(BaseModel):
    start_latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
    )

    start_longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
    )

    goal_latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
    )

    goal_longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
    )

    max_distance_km: float = Field(
        default=30.0,
        gt=0.0,
        le=100.0,
    )

    max_expanded_nodes: int = Field(
        default=100000,
        gt=100,
        le=500000,
    )


class EvacuationRequest(LocationRequest):
    forecast_sample: int = Field(
        default=0,
        ge=0,
    )

    forecast_day: int = Field(
        default=1,
        ge=0,
        le=6,
    )

    candidate_shelters: int = Field(
        default=12,
        ge=1,
        le=24,
    )


class ShelterSummary(BaseModel):
    shelter_id: int
    latitude: float
    longitude: float
    road_node: int | None = None
    snap_distance_m: float | None = None
    mapped: bool


class SheltersResponse(BaseModel):
    total_shelters: int
    mapped_shelters: int
    unmapped_shelters: int
    shelters: list[ShelterSummary]


class HazardResponse(BaseModel):
    latitude: float
    longitude: float
    forecast_sample: int
    forecast_day: int

    grid_latitude: float
    grid_longitude: float

    hazard_score: float
    population_exposure: float | None = None
    population_component: float | None = None
    population_density: float | None = None

    sampling_method: str
    grid_distance_degrees: float | None = None


class RouteResponse(BaseModel):
    result: dict[str, Any]


class EvacuationResponse(BaseModel):
    result: dict[str, Any]
