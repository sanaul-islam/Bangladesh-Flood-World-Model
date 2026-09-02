from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class ForecastResponse(BaseModel):
    forecast_sample: int
    forecast_days: list[int]

    latitude_points: int
    longitude_points: int

    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hazard
#
# IMPORTANT:
# The forecasting system uses human-readable forecast days:
# Day 1 ... Day 7
#
# Therefore the API contract MUST be 1 <= forecast_day <= 7.
# ---------------------------------------------------------------------------

class HazardRequest(LocationRequest):
    forecast_sample: int = Field(
        default=0,
        ge=0,
    )
    forecast_day: int = Field(
        default=1,
        ge=1,
        le=7,
    )


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


# ---------------------------------------------------------------------------
# Shelters
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

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
        default=100_000,
        gt=100,
        le=500_000,
    )


class RouteStatistics(BaseModel):
    road_edges: int
    road_distance_km: float
    estimated_travel_time_min: float

    risk_cost: float

    mean_flood_risk: float
    maximum_flood_risk: float

    mean_uncertainty_risk: float
    maximum_uncertainty_risk: float

    mean_total_risk: float

    maximum_bridge_risk: float
    bridge_edges: int


class RouteGeometry(BaseModel):
    nodes: list[int]
    edge_ids: list[int]

    # [longitude, latitude]
    coordinates: list[tuple[float, float]]


class RouteEndpoint(BaseModel):
    latitude: float
    longitude: float

    nearest_node: int
    snap_distance_m: float


class RouteRoutingMetadata(BaseModel):
    algorithm: str
    expanded_nodes: int
    max_search_distance_km: float


class RoutePayload(BaseModel):
    start: RouteEndpoint
    goal: RouteEndpoint

    routing: RouteRoutingMetadata

    route: RouteGeometry

    statistics: RouteStatistics


class RouteResponse(BaseModel):
    status: str
    result: RoutePayload


# ---------------------------------------------------------------------------
# Evacuation
# ---------------------------------------------------------------------------

class EvacuationRequest(LocationRequest):
    forecast_sample: int = Field(
        default=0,
        ge=0,
    )

    forecast_day: int = Field(
        default=1,
        ge=1,
        le=7,
    )

    candidate_shelters: int = Field(
        default=12,
        ge=1,
        le=24,
    )


class EvacuationSystemInfo(BaseModel):
    name: str

    forecast_sample: int
    forecast_day: int

    candidate_shelters: int
    reachable_shelters: int

    total_mapped_shelters: int

    database_mode: str

    shelter_capacity_available: bool
    live_shelter_availability_available: bool


class EvacuationUserInfo(BaseModel):
    latitude: float
    longitude: float

    nearest_road_node: int
    snap_distance_m: float


class RankingWeights(BaseModel):
    route_risk: float
    destination_hazard: float
    destination_population_exposure: float
    travel_time: float
    bridge_exposure: float


class ShelterRecommendation(BaseModel):
    shelter_id: int

    latitude: float
    longitude: float

    road_node: int

    snap_distance_m: float

    straight_distance_km: float
    road_distance_km: float
    travel_time_min: float

    risk_cost: float

    mean_flood_risk: float
    maximum_flood_risk: float

    mean_uncertainty: float
    maximum_uncertainty: float

    maximum_bridge_risk: float
    bridge_edges: int

    destination_hazard: float
    destination_population_exposure: float
    destination_population_component: float
    destination_population_density: float

    availability: str
    capacity: int | None = None

    source: str

    route_risk_score: float
    travel_time_score: float
    bridge_score: float
    accessibility_score: float

    combined_score: float
    rank: int


class EvacuationResult(BaseModel):
    system: EvacuationSystemInfo

    user: EvacuationUserInfo

    ranking_weights: RankingWeights

    recommended_shelter: ShelterRecommendation

    route: RoutePayload

    alternatives: list[ShelterRecommendation]


class EvacuationResponse(BaseModel):
    status: str
    result: EvacuationResult
