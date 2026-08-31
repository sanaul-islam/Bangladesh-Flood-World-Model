from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import xarray as xr
from shapely.geometry import Point


class EvacuationPlanner:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

        self.roads_path = self.project_root / "data/static/roads/roads.shp"
        self.bridges_path = self.project_root / "data/static/bridges/bridges.shp"
        self.shelters_path = self.project_root / "data/static/shelters/shelters.shp"

    @staticmethod
    def nearest_grid_value(hazard, lat, lon):
        lat_index = int(np.argmin(np.abs(hazard.lat.values - lat)))
        lon_index = int(np.argmin(np.abs(hazard.lon.values - lon)))

        return float(
            hazard.values[
                lat_index,
                lon_index,
            ]
        )

    @staticmethod
    def nearest_node(graph, point):
        nodes = list(graph.nodes)

        return min(
            nodes,
            key=lambda node: (
                node[0] - point.x
            ) ** 2
            + (
                node[1] - point.y
            ) ** 2,
        )

    def attach_hazard(self, graph, hazard):
        for u, v, edge in graph.edges(data=True):
            geometry = edge["geometry"]

            midpoint = geometry.interpolate(
                0.5,
                normalized=True,
            )

            risk = self.nearest_grid_value(
                hazard,
                midpoint.y,
                midpoint.x,
            )

            edge["hazard"] = risk

            length = edge["length_m"]

            travel_cost = length

            hazard_penalty = 1.0 + 10.0 * risk

            edge["weight"] = (
                travel_cost
                * hazard_penalty
            )

    def plan(self, graph, hazard, start_lat, start_lon, forecast_day=1):
        selected_hazard = hazard.sel(
            forecast_day=forecast_day
        )

        self.attach_hazard(
            graph,
            selected_hazard,
        )

        start_point = Point(
            start_lon,
            start_lat,
        )

        start_node = self.nearest_node(
            graph,
            start_point,
        )

        shelters = gpd.read_file(
            self.shelters_path
        ).to_crs("EPSG:4326")

        candidates = []

        for shelter_index, row in shelters.iterrows():
            geometry = row.geometry

            if geometry is None or geometry.is_empty:
                continue

            shelter_node = self.nearest_node(
                graph,
                geometry,
            )

            try:
                path = nx.shortest_path(
                    graph,
                    start_node,
                    shelter_node,
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                continue

            distance_m = 0.0
            hazard_sum = 0.0

            for u, v in zip(
                path[:-1],
                path[1:],
            ):
                edge = graph[u][v]

                distance_m += edge["length_m"]
                hazard_sum += edge["hazard"]

            edge_count = max(
                len(path) - 1,
                1,
            )

            mean_hazard = (
                hazard_sum
                / edge_count
            )

            score = (
                distance_m
                * (
                    1.0
                    + 10.0 * mean_hazard
                )
            )

            candidates.append(
                {
                    "shelter_index": int(shelter_index),
                    "path": path,
                    "distance_m": distance_m,
                    "mean_hazard": mean_hazard,
                    "score": score,
                }
            )

        if not candidates:
            raise RuntimeError("No reachable shelter found.")

        candidates.sort(
            key=lambda x: x["score"]
        )

        return candidates[0], candidates

    @staticmethod
    def save_route(route, output_path: Path):
        coordinates = [
            [lon, lat]
            for lon, lat in route["path"]
        ]

        gdf = gpd.GeoDataFrame(
            {
                "order": list(
                    range(
                        len(
                            coordinates
                        )
                    )
                )
            },
            geometry=[
                Point(
                    lon,
                    lat,
                )
                for lon, lat in coordinates
            ],
            crs="EPSG:4326",
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        gdf.to_file(
            output_path,
            driver="GeoJSON",
        )

        print(f"Saved route: {output_path}")