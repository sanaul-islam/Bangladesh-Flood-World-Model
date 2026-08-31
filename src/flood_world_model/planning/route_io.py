from __future__ import annotations

import json
from pathlib import Path


def save_route_geojson(
    route_result: dict,
    output_path: str | Path,
) -> None:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    route = route_result[
        "route"
    ]

    coordinates = route[
        "coordinates"
    ]

    statistics = route_result[
        "statistics"
    ]

    feature = {
        "type": "Feature",
        "properties": {
            "road_edges": int(
                statistics[
                    "road_edges"
                ]
            ),
            "road_distance_km": float(
                statistics[
                    "road_distance_km"
                ]
            ),
            "estimated_travel_time_min": float(
                statistics[
                    "estimated_travel_time_min"
                ]
            ),
            "risk_cost": float(
                statistics[
                    "risk_cost"
                ]
            ),
            "mean_flood_risk": float(
                statistics[
                    "mean_flood_risk"
                ]
            ),
            "maximum_flood_risk": float(
                statistics[
                    "maximum_flood_risk"
                ]
            ),
            "mean_uncertainty_risk": float(
                statistics[
                    "mean_uncertainty_risk"
                ]
            ),
            "maximum_uncertainty_risk": float(
                statistics[
                    "maximum_uncertainty_risk"
                ]
            ),
            "maximum_bridge_risk": float(
                statistics[
                    "maximum_bridge_risk"
                ]
            ),
            "bridge_edges": int(
                statistics[
                    "bridge_edges"
                ]
            ),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
    }

    collection = {
        "type": "FeatureCollection",
        "features": [
            feature
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            collection,
            file,
            indent=2,
        )


def save_route_metrics(
    route_result: dict,
    output_path: str | Path,
) -> None:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            route_result,
            file,
            indent=2,
        )
