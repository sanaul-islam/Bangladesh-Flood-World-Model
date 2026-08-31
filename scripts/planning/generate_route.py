from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import xarray as xr

from flood_world_model.planning.graph import RoadGraph
from flood_world_model.planning.planner import EvacuationPlanner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


START_LAT = 23.8103
START_LON = 90.4125


def main():
    print("=" * 80)
    print("RISK-AWARE EVACUATION PLANNER")
    print("=" * 80)

    print(f"User location: {START_LAT}, {START_LON}")

    road_graph_builder = RoadGraph(
        PROJECT_ROOT / "data/static/roads/roads.shp",
        PROJECT_ROOT / "data/static/bridges/bridges.shp",
    )

    graph = road_graph_builder.build(
        allowed_highways=[
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "unclassified",
        ]
    )

    print("Road graph built.")

    hazard_ds = xr.open_dataset(
        PROJECT_ROOT / "outputs/hazard/v0_7day_hazard.nc"
    )

    planner = EvacuationPlanner(
        PROJECT_ROOT
    )

    best, candidates = planner.plan(
        graph,
        hazard_ds["hazard_score"],
        START_LAT,
        START_LON,
        forecast_day=1,
    )

    print("=" * 80)
    print("BEST ROUTE")
    print("=" * 80)

    print(f"Shelter index: {best['shelter_index']}")
    print(f"Distance: {best['distance_m'] / 1000.0:.2f} km")
    print(f"Mean hazard: {best['mean_hazard']:.4f}")
    print(f"Route score: {best['score']:.2f}")

    print("=" * 80)
    print("TOP ALTERNATIVES")
    print("=" * 80)

    for rank, candidate in enumerate(
        candidates[:5],
        start=1,
    ):
        print(
            f"{rank}. shelter={candidate['shelter_index']} "
            f"distance={candidate['distance_m'] / 1000.0:.2f}km "
            f"hazard={candidate['mean_hazard']:.4f} "
            f"score={candidate['score']:.2f}"
        )

    route_path = (
        PROJECT_ROOT
        / "outputs/routes/best_route.geojson"
    )

    planner.save_route(
        best,
        route_path,
    )

    hazard_ds.close()

    print("✅ Route planning complete.")


if __name__ == "__main__":
    main()