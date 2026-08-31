from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from flood_world_model.planning.routing import (
    RoadRouter,
)

from flood_world_model.planning.shelter_ranking import (
    ShelterRanker,
)


class FloodWorldModelService:

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

        self.ranker = ShelterRanker(
            database_path=self.database_path,
            population_risk_path=self.population_risk_path,
        )

        self.router = self.ranker.router

        self.read_connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            timeout=30.0,
        )

        self.read_connection.execute(
            "PRAGMA query_only = ON"
        )

        self.read_connection.execute(
            "PRAGMA cache_size = -32768"
        )

        self.ready = True

    def close(
        self,
    ) -> None:

        if not self.ready:
            return

        self.ready = False

        try:
            self.read_connection.close()
        finally:
            self.ranker.close()

    def _ensure_ready(
        self,
    ) -> None:

        if not self.ready:
            raise RuntimeError(
                "Flood World Model service is not ready."
            )

    def health(
        self,
    ) -> dict[str, Any]:

        self._ensure_ready()

        return {
            "status": "ok",
            "database_exists": (
                self.database_path.exists()
            ),
            "population_risk_exists": (
                self.population_risk_path.exists()
            ),
        }

    def forecast_metadata(
        self,
    ) -> dict[str, Any]:

        self._ensure_ready()

        ds = self.ranker.risk_ds

        lat = np.asarray(
            ds[
                "lat"
            ].values,
            dtype=np.float64,
        )

        lon = np.asarray(
            ds[
                "lon"
            ].values,
            dtype=np.float64,
        )

        samples = [
            int(value)
            for value in ds[
                "sample"
            ].values
        ]

        days = [
            int(value)
            for value in ds[
                "forecast_day"
            ].values
        ]

        return {
            "forecast_sample": (
                samples[0]
                if samples
                else 0
            ),
            "forecast_samples": samples,
            "forecast_days": days,
            "latitude_points": int(
                len(lat)
            ),
            "longitude_points": int(
                len(lon)
            ),
            "latitude_min": float(
                lat.min()
            ),
            "latitude_max": float(
                lat.max()
            ),
            "longitude_min": float(
                lon.min()
            ),
            "longitude_max": float(
                lon.max()
            ),
        }

    def hazard(
        self,
        latitude: float,
        longitude: float,
        forecast_sample: int,
        forecast_day: int,
    ) -> dict[str, Any]:

        self._ensure_ready()

        values = (
            self.ranker.destination_values(
                latitude=latitude,
                longitude=longitude,
                forecast_sample=forecast_sample,
                forecast_day=forecast_day,
            )
        )

        sampling = values[
            "hazard_sampling"
        ]

        lat_index = sampling.get(
            "lat_index"
        )

        if lat_index is None:
            lat_index = sampling.get(
                "lat_index_0"
            )

        lon_index = sampling.get(
            "lon_index"
        )

        if lon_index is None:
            lon_index = sampling.get(
                "lon_index_0"
            )

        return {
            "latitude": float(
                latitude
            ),
            "longitude": float(
                longitude
            ),
            "forecast_sample": int(
                forecast_sample
            ),
            "forecast_day": int(
                forecast_day
            ),
            "grid_latitude": float(
                self.ranker.lat_values[
                    int(lat_index)
                ]
            ),
            "grid_longitude": float(
                self.ranker.lon_values[
                    int(lon_index)
                ]
            ),
            "hazard_score": float(
                values[
                    "hazard_score"
                ]
            ),
            "population_exposure": values[
                "population_exposure"
            ],
            "population_component": values[
                "population_component"
            ],
            "population_density": values[
                "population_density"
            ],
            "sampling_method": values[
                "hazard_sampling"
            ][
                "method"
            ],
        }

    def shelters(
        self,
    ) -> dict[str, Any]:

        self._ensure_ready()

        rows = self.read_connection.execute(
            """
            SELECT
                s.shelter_id,
                s.x,
                s.y,
                m.node_id,
                m.distance_m
            FROM shelters AS s
            LEFT JOIN shelter_node_map AS m
                ON m.shelter_id = s.shelter_id
            ORDER BY s.shelter_id
            """
        ).fetchall()

        shelters = []

        for row in rows:

            (
                shelter_id,
                x,
                y,
                node_id,
                distance_m,
            ) = row

            latitude, longitude = (
                self.router.xy_to_latlon(
                    float(x),
                    float(y),
                )
            )

            shelters.append(
                {
                    "shelter_id": int(
                        shelter_id
                    ),
                    "latitude": float(
                        latitude
                    ),
                    "longitude": float(
                        longitude
                    ),
                    "road_node": (
                        int(node_id)
                        if node_id is not None
                        else None
                    ),
                    "snap_distance_m": (
                        float(distance_m)
                        if distance_m is not None
                        else None
                    ),
                    "mapped": (
                        node_id is not None
                    ),
                }
            )

        total = len(
            shelters
        )

        mapped = sum(
            shelter[
                "mapped"
            ]
            for shelter in shelters
        )

        return {
            "total_shelters": int(
                total
            ),
            "mapped_shelters": int(
                mapped
            ),
            "unmapped_shelters": int(
                total - mapped
            ),
            "shelters": shelters,
        }

    def route(
        self,
        start_latitude: float,
        start_longitude: float,
        goal_latitude: float,
        goal_longitude: float,
        max_distance_km: float,
        max_expanded_nodes: int,
    ) -> dict[str, Any]:

        self._ensure_ready()

        return self.router.route(
            start_latitude=start_latitude,
            start_longitude=start_longitude,
            goal_latitude=goal_latitude,
            goal_longitude=goal_longitude,
            max_distance_km=max_distance_km,
            max_expanded_nodes=max_expanded_nodes,
        )

    def evacuate(
        self,
        latitude: float,
        longitude: float,
        forecast_sample: int,
        forecast_day: int,
        candidate_shelters: int,
    ) -> dict[str, Any]:

        self._ensure_ready()

        return self.ranker.rank(
            user_latitude=latitude,
            user_longitude=longitude,
            forecast_sample=forecast_sample,
            forecast_day=forecast_day,
            candidate_count=candidate_shelters,
        )
