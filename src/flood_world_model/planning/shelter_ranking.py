from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from flood_world_model.planning.grid_sampling import (
    bilinear_sample,
)
from flood_world_model.planning.routing import (
    RoadRouter,
    RouteNotFoundError,
)


DEFAULT_CANDIDATE_COUNT = 24
DEFAULT_ROUTE_EXPANSION_LIMIT = 100000

ROUTE_RISK_WEIGHT = 0.35
DESTINATION_HAZARD_WEIGHT = 0.20
DESTINATION_EXPOSURE_WEIGHT = 0.20
TRAVEL_TIME_WEIGHT = 0.15
BRIDGE_WEIGHT = 0.10


class ShelterRanker:

    def __init__(
        self,
        database_path: str | Path,
        population_risk_path: str | Path,
    ) -> None:

        self.database_path = Path(
            database_path
        )

        self.population_risk_path = Path(
            population_risk_path
        )

        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Missing database: {self.database_path}"
            )

        if not self.population_risk_path.exists():
            raise FileNotFoundError(
                f"Missing population-risk file: "
                f"{self.population_risk_path}"
            )

        self._lock = threading.RLock()

        self.connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=30.0,
        )

        self.connection.execute(
            "PRAGMA query_only = ON"
        )

        self.connection.execute(
            "PRAGMA cache_size = -65536"
        )

        self.router = RoadRouter(
            self.database_path
        )

        self.risk_ds = xr.open_dataset(
            self.population_risk_path
        )

        self.samples = (
            self.risk_ds[
                "sample"
            ].values
        )

        self.forecast_days = (
            self.risk_ds[
                "forecast_day"
            ].values
        )

        self.lat_values = (
            self.risk_ds[
                "lat"
            ].values
        )

        self.lon_values = (
            self.risk_ds[
                "lon"
            ].values
        )

        self.hazard_data = (
            self.risk_ds[
                "hydrological_hazard_score"
            ]
            .values
            .astype(
                np.float32
            )
        )

        self.exposure_data = (
            self.risk_ds[
                "population_exposure_index"
            ]
            .values
            .astype(
                np.float32
            )
        )

        self.population_component_data = (
            self.risk_ds[
                "population_component"
            ]
            .values
            .astype(
                np.float32
            )
            if "population_component"
            in self.risk_ds
            else None
        )

        self.population_density_data = (
            self.risk_ds[
                "population_density"
            ]
            .values
            .astype(
                np.float32
            )
            if "population_density"
            in self.risk_ds
            else None
        )

        self.closed = False

    def close(
        self,
    ) -> None:

        with self._lock:

            if self.closed:
                return

            self.closed = True

            try:
                self.router.close()
            finally:
                try:
                    self.risk_ds.close()
                finally:
                    self.connection.close()

    def __enter__(
        self,
    ) -> "ShelterRanker":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def _ensure_open(
        self,
    ) -> None:

        if self.closed:
            raise RuntimeError(
                "ShelterRanker is closed."
            )

    def _positions(
        self,
        sample: int,
        day: int,
    ) -> tuple[int, int]:

        sample_match = np.where(
            self.samples == sample
        )[0]

        day_match = np.where(
            self.forecast_days == day
        )[0]

        if len(sample_match) == 0:
            raise ValueError(
                f"Unknown forecast sample: {sample}"
            )

        if len(day_match) == 0:
            raise ValueError(
                f"Unknown forecast day: {day}"
            )

        return (
            int(sample_match[0]),
            int(day_match[0]),
        )

    def destination_values(
        self,
        latitude: float,
        longitude: float,
        forecast_sample: int,
        forecast_day: int,
    ) -> dict:

        with self._lock:

            self._ensure_open()

            sample_index, day_index = (
                self._positions(
                    forecast_sample,
                    forecast_day,
                )
            )

            hazard = self.hazard_data[
                sample_index,
                day_index,
            ]

            exposure = self.exposure_data[
                sample_index,
                day_index,
            ]

            hazard_value, hazard_meta = (
                bilinear_sample(
                    hazard,
                    self.lat_values,
                    self.lon_values,
                    latitude,
                    longitude,
                )
            )

            exposure_value, exposure_meta = (
                bilinear_sample(
                    exposure,
                    self.lat_values,
                    self.lon_values,
                    latitude,
                    longitude,
                )
            )

            if hazard_value is None:
                raise RuntimeError(
                    "No valid hydrological hazard value "
                    "is available near the requested location."
                )

            if exposure_value is None:
                raise RuntimeError(
                    "No valid population exposure value "
                    "is available near the requested location."
                )

            component_value = None

            if self.population_component_data is not None:

                component_value, _ = bilinear_sample(
                    self.population_component_data[
                        sample_index,
                        day_index,
                    ],
                    self.lat_values,
                    self.lon_values,
                    latitude,
                    longitude,
                )

            density_value = None

            if self.population_density_data is not None:

                density_value, _ = bilinear_sample(
                    self.population_density_data[
                        sample_index,
                        day_index,
                    ],
                    self.lat_values,
                    self.lon_values,
                    latitude,
                    longitude,
                )

            if component_value is not None:
                component_value = float(
                    np.clip(
                        component_value,
                        0.0,
                        1.0,
                    )
                )

            if density_value is not None:
                density_value = max(
                    float(
                        density_value
                    ),
                    0.0,
                )

            return {
                "hazard_score": float(
                    np.clip(
                        hazard_value,
                        0.0,
                        1.0,
                    )
                ),
                "population_exposure": float(
                    np.clip(
                        exposure_value,
                        0.0,
                        1.0,
                    )
                ),
                "population_component": component_value,
                "population_density": density_value,
                "hazard_sampling": hazard_meta,
                "exposure_sampling": exposure_meta,
            }

    def _candidates(
        self,
        user_x: float,
        user_y: float,
        limit: int,
    ) -> list[dict]:

        self._ensure_open()

        rows = self.connection.execute(
            """
            SELECT
                s.shelter_id,
                s.x,
                s.y,
                m.node_id,
                m.distance_m,
                (
                    (s.x - ?) * (s.x - ?)
                    +
                    (s.y - ?) * (s.y - ?)
                ) AS d2
            FROM shelters AS s
            JOIN shelter_node_map AS m
                ON m.shelter_id = s.shelter_id
            ORDER BY d2
            LIMIT ?
            """,
            (
                user_x,
                user_x,
                user_y,
                user_y,
                limit,
            ),
        ).fetchall()

        return [
            {
                "shelter_id": int(
                    row[0]
                ),
                "x": float(
                    row[1]
                ),
                "y": float(
                    row[2]
                ),
                "road_node": int(
                    row[3]
                ),
                "snap_distance_m": float(
                    row[4]
                ),
                "straight_distance_km": (
                    math.sqrt(
                        float(
                            row[5]
                        )
                    )
                    / 1000.0
                ),
            }
            for row in rows
        ]

    @staticmethod
    def _normalize(
        values: list[float],
    ) -> list[float]:

        if not values:
            return []

        low = min(
            values
        )

        high = max(
            values
        )

        if high <= low:
            return [
                0.0
                for _ in values
            ]

        return [
            float(
                (
                    value - low
                )
                / (
                    high - low
                )
            )
            for value in values
        ]

    def _rank_locked(
        self,
        user_latitude: float,
        user_longitude: float,
        forecast_sample: int,
        forecast_day: int,
        candidate_count: int,
    ) -> dict:

        self._ensure_open()

        (
            user_node,
            user_x,
            user_y,
            user_snap_distance,
        ) = self.router.nearest_node(
            user_latitude,
            user_longitude,
        )

        candidates = self._candidates(
            user_x=user_x,
            user_y=user_y,
            limit=candidate_count,
        )

        if not candidates:
            raise RouteNotFoundError(
                "No mapped shelters found."
            )

        evaluated = []

        for shelter in candidates:

            latitude, longitude = (
                self.router.xy_to_latlon(
                    shelter[
                        "x"
                    ],
                    shelter[
                        "y"
                    ],
                )
            )

            destination = (
                self.destination_values(
                    latitude=latitude,
                    longitude=longitude,
                    forecast_sample=forecast_sample,
                    forecast_day=forecast_day,
                )
            )

            route = self.router.route(
                start_latitude=user_latitude,
                start_longitude=user_longitude,
                goal_latitude=latitude,
                goal_longitude=longitude,
                max_distance_km=30.0,
                max_expanded_nodes=DEFAULT_ROUTE_EXPANSION_LIMIT,
            )

            stats = route[
                "statistics"
            ]

            evaluated.append(
                {
                    "shelter_id": shelter[
                        "shelter_id"
                    ],
                    "latitude": float(
                        latitude
                    ),
                    "longitude": float(
                        longitude
                    ),
                    "road_node": shelter[
                        "road_node"
                    ],
                    "snap_distance_m": shelter[
                        "snap_distance_m"
                    ],
                    "straight_distance_km": shelter[
                        "straight_distance_km"
                    ],
                    "route": route,
                    "road_distance_km": float(
                        stats[
                            "road_distance_km"
                        ]
                    ),
                    "travel_time_min": float(
                        stats[
                            "estimated_travel_time_min"
                        ]
                    ),
                    "risk_cost": float(
                        stats[
                            "risk_cost"
                        ]
                    ),
                    "mean_flood_risk": float(
                        stats[
                            "mean_flood_risk"
                        ]
                    ),
                    "maximum_flood_risk": float(
                        stats[
                            "maximum_flood_risk"
                        ]
                    ),
                    "mean_uncertainty": float(
                        stats[
                            "mean_uncertainty_risk"
                        ]
                    ),
                    "maximum_uncertainty": float(
                        stats[
                            "maximum_uncertainty_risk"
                        ]
                    ),
                    "maximum_bridge_risk": float(
                        stats[
                            "maximum_bridge_risk"
                        ]
                    ),
                    "bridge_edges": int(
                        stats[
                            "bridge_edges"
                        ]
                    ),
                    "destination_hazard": float(
                        destination[
                            "hazard_score"
                        ]
                    ),
                    "destination_population_exposure": float(
                        destination[
                            "population_exposure"
                        ]
                    ),
                    "destination_population_component": (
                        destination[
                            "population_component"
                        ]
                    ),
                    "destination_population_density": (
                        destination[
                            "population_density"
                        ]
                    ),
                    "destination_hazard_sampling": destination[
                        "hazard_sampling"
                    ],
                    "destination_exposure_sampling": destination[
                        "exposure_sampling"
                    ],
                    "availability": "unknown",
                    "capacity": None,
                    "source": "OSM",
                }
            )

        if not evaluated:
            raise RouteNotFoundError(
                "No reachable shelter had valid forecast coverage."
            )

        risk_scores = self._normalize(
            [
                item[
                    "risk_cost"
                ]
                for item in evaluated
            ]
        )

        time_scores = self._normalize(
            [
                item[
                    "travel_time_min"
                ]
                for item in evaluated
            ]
        )

        bridge_scores = self._normalize(
            [
                float(
                    item[
                        "bridge_edges"
                    ]
                )
                for item in evaluated
            ]
        )

        for i, item in enumerate(
            evaluated
        ):

            item[
                "route_risk_score"
            ] = float(
                risk_scores[i]
            )

            item[
                "travel_time_score"
            ] = float(
                time_scores[i]
            )

            item[
                "bridge_score"
            ] = float(
                bridge_scores[i]
            )

            item[
                "accessibility_score"
            ] = float(
                1.0 - risk_scores[i]
            )

            item[
                "combined_score"
            ] = float(
                ROUTE_RISK_WEIGHT
                * risk_scores[i]
                + DESTINATION_HAZARD_WEIGHT
                * item[
                    "destination_hazard"
                ]
                + DESTINATION_EXPOSURE_WEIGHT
                * item[
                    "destination_population_exposure"
                ]
                + TRAVEL_TIME_WEIGHT
                * time_scores[i]
                + BRIDGE_WEIGHT
                * bridge_scores[i]
            )

        evaluated.sort(
            key=lambda item: (
                item[
                    "combined_score"
                ],
                item[
                    "travel_time_min"
                ],
            )
        )

        for rank, item in enumerate(
            evaluated,
            start=1,
        ):
            item[
                "rank"
            ] = rank

        best = evaluated[
            0
        ]

        return {
            "system": {
                "name": (
                    "Bangladesh Flood World Model "
                    "Risk-Aware Evacuation System"
                ),
                "forecast_sample": int(
                    forecast_sample
                ),
                "forecast_day": int(
                    forecast_day
                ),
                "candidate_shelters": int(
                    candidate_count
                ),
                "reachable_shelters": int(
                    len(evaluated)
                ),
                "total_mapped_shelters": int(
                    self.connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM shelter_node_map
                        """
                    ).fetchone()[0]
                ),
                "database_mode": "read_only",
                "shelter_capacity_available": False,
                "live_shelter_availability_available": False,
            },
            "user": {
                "latitude": float(
                    user_latitude
                ),
                "longitude": float(
                    user_longitude
                ),
                "nearest_road_node": int(
                    user_node
                ),
                "snap_distance_m": float(
                    user_snap_distance
                ),
            },
            "ranking_weights": {
                "route_risk": ROUTE_RISK_WEIGHT,
                "destination_hazard": (
                    DESTINATION_HAZARD_WEIGHT
                ),
                "destination_population_exposure": (
                    DESTINATION_EXPOSURE_WEIGHT
                ),
                "travel_time": TRAVEL_TIME_WEIGHT,
                "bridge_exposure": BRIDGE_WEIGHT,
            },
            "recommended_shelter": {
                key: value
                for key, value in best.items()
                if key != "route"
            },
            "route": best[
                "route"
            ],
            "alternatives": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "route"
                }
                for item in evaluated[
                    1:
                ]
            ],
        }

    def rank(
        self,
        user_latitude: float,
        user_longitude: float,
        forecast_sample: int,
        forecast_day: int,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    ) -> dict:

        with self._lock:

            return self._rank_locked(
                user_latitude=user_latitude,
                user_longitude=user_longitude,
                forecast_sample=forecast_sample,
                forecast_day=forecast_day,
                candidate_count=candidate_count,
            )
