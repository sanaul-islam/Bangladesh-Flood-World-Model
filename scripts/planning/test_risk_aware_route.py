from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT / "src"
    ),
)

from flood_world_model.planning.routing import (
    RoadRouter,
)

from flood_world_model.planning.route_io import (
    save_route_geojson,
    save_route_metrics,
)


DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

GEOJSON_PATH = (
    PROJECT_ROOT
    / "outputs/routes/test_risk_aware_route.geojson"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/routes/test_risk_aware_route.json"
)


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("LOCAL RISK-AWARE ROUTING TEST")
    print("=" * 80)

    print(
        f"Database: {DATABASE_PATH}"
    )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing road database: {DATABASE_PATH}"
        )

    # Test coordinates.
    #
    # Replace these with actual user GPS coordinates and a target shelter
    # coordinate once shelter routing is connected.
    start_latitude = 23.8103
    start_longitude = 90.4125

    goal_latitude = 23.7500
    goal_longitude = 90.3900

    print(
        f"Start: {start_latitude}, {start_longitude}"
    )

    print(
        f"Goal: {goal_latitude}, {goal_longitude}"
    )

    with RoadRouter(
        DATABASE_PATH
    ) as router:

        result = router.route(
            start_latitude=start_latitude,
            start_longitude=start_longitude,
            goal_latitude=goal_latitude,
            goal_longitude=goal_longitude,
            initial_radius_km=5.0,
            radius_increment_km=5.0,
            max_radius_km=30.0,
        )

    save_route_geojson(
        route_result=result,
        output_path=GEOJSON_PATH,
    )

    save_route_metrics(
        route_result=result,
        output_path=METRICS_PATH,
    )

    statistics = (
        result[
            "statistics"
        ]
    )

    print("=" * 80)
    print("RISK-AWARE ROUTE COMPLETE")
    print("=" * 80)

    print(
        f"Route nodes: {len(result['route']['nodes']):,}"
    )

    print(
        f"Road distance: {statistics['road_distance_km']:.2f} km"
    )

    print(
        f"Estimated travel time: {statistics['estimated_travel_time_min']:.2f} min"
    )

    print(
        f"Risk cost: {statistics['risk_cost']:.2f}"
    )

    print(
        f"Maximum flood risk: {statistics['maximum_flood_risk']:.4f}"
    )

    print(
        f"Maximum uncertainty risk: {statistics['maximum_uncertainty_risk']:.4f}"
    )

    print(
        f"Maximum bridge risk: {statistics['maximum_bridge_risk']:.4f}"
    )

    print(
        f"Bridge edges: {statistics['bridge_edges']:,}"
    )

    print(
        f"GeoJSON: {GEOJSON_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
