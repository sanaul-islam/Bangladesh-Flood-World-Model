from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString


DEFAULT_SPEED_KMH = 30.0
CHUNK_SIZE = 10000
COORDINATE_PRECISION = 1.0


def safe_float(
    value: Any,
    default: float,
) -> float:
    try:
        value = float(value)

        if not math.isfinite(
            value
        ):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


def estimate_speed_kmh(
    row: pd.Series,
) -> float:
    for column in [
        "maxspeed",
        "speed_kmh",
        "speed",
    ]:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        if isinstance(
            value,
            str,
        ):
            text = value.strip()

            if ";" in text:
                text = text.split(
                    ";"
                )[0]

            digits = ""

            for character in text:
                if (
                    character.isdigit()
                    or character == "."
                ):
                    digits += character

            if digits:
                parsed = safe_float(
                    digits,
                    DEFAULT_SPEED_KMH,
                )

                if 5.0 <= parsed <= 150.0:
                    return parsed

        else:
            parsed = safe_float(
                value,
                DEFAULT_SPEED_KMH,
            )

            if 5.0 <= parsed <= 150.0:
                return parsed

    highway = str(
        row.get(
            "highway",
            "",
        )
    ).lower()

    highway_speed = {
        "motorway": 80.0,
        "motorway_link": 60.0,
        "trunk": 70.0,
        "trunk_link": 55.0,
        "primary": 55.0,
        "primary_link": 45.0,
        "secondary": 45.0,
        "secondary_link": 40.0,
        "tertiary": 40.0,
        "tertiary_link": 35.0,
        "unclassified": 30.0,
        "residential": 25.0,
        "service": 15.0,
        "track": 10.0,
        "path": 5.0,
        "footway": 5.0,
        "pedestrian": 5.0,
    }

    return highway_speed.get(
        highway,
        DEFAULT_SPEED_KMH,
    )


def normalize_line_geometry(
    geometry: Any,
) -> list[LineString]:
    if geometry is None:
        return []

    if geometry.is_empty:
        return []

    geometry_type = geometry.geom_type

    if geometry_type == "LineString":
        return [
            geometry
        ]

    if geometry_type == "MultiLineString":
        return [
            line
            for line in geometry.geoms
            if not line.is_empty
        ]

    return []


def node_key(
    x: float,
    y: float,
) -> tuple[int, int]:
    return (
        int(
            round(
                x
                / COORDINATE_PRECISION
            )
        ),
        int(
            round(
                y
                / COORDINATE_PRECISION
            )
        ),
    )


def node_xy(
    node: tuple[int, int],
) -> tuple[float, float]:
    return (
        node[0]
        * COORDINATE_PRECISION,
        node[1]
        * COORDINATE_PRECISION,
    )


def process_chunk(
    roads: gpd.GeoDataFrame,
    edge_records: list[dict[str, Any]],
    next_edge_id: int,
) -> int:
    for feature_index, row in roads.iterrows():

        geometries = normalize_line_geometry(
            row.geometry
        )

        if not geometries:
            continue

        speed_kmh = estimate_speed_kmh(
            row
        )

        highway = str(
            row.get(
                "highway",
                "",
            )
        )

        for geometry in geometries:

            coordinates = list(
                geometry.coords
            )

            if len(coordinates) < 2:
                continue

            for point_index in range(
                len(coordinates) - 1
            ):
                x1, y1 = coordinates[
                    point_index
                ]

                x2, y2 = coordinates[
                    point_index + 1
                ]

                u = node_key(
                    float(x1),
                    float(y1),
                )

                v = node_key(
                    float(x2),
                    float(y2),
                )

                if u == v:
                    continue

                ux, uy = node_xy(
                    u
                )

                vx, vy = node_xy(
                    v
                )

                length_m = math.hypot(
                    vx - ux,
                    vy - uy,
                )

                if (
                    not math.isfinite(
                        length_m
                    )
                    or length_m <= 0.0
                ):
                    continue

                travel_time_s = (
                    length_m
                    / 1000.0
                    / max(
                        speed_kmh,
                        1.0,
                    )
                    * 3600.0
                )

                edge_records.append(
                    {
                        "edge_id": next_edge_id,
                        "u_x": ux,
                        "u_y": uy,
                        "v_x": vx,
                        "v_y": vy,
                        "road_feature_id": str(
                            feature_index
                        ),
                        "highway": highway,
                        "length_m": float(
                            length_m
                        ),
                        "speed_kmh": float(
                            speed_kmh
                        ),
                        "travel_time_s": float(
                            travel_time_s
                        ),
                    }
                )

                next_edge_id += 1

    return next_edge_id


def build_road_edges(
    roads_path: str | Path,
    output_crs: str = "EPSG:32646",
    chunk_size: int = CHUNK_SIZE,
) -> gpd.GeoDataFrame:
    roads_path = Path(
        roads_path
    )

    if not roads_path.exists():
        raise FileNotFoundError(
            f"Road file not found: {roads_path}"
        )

    roads = gpd.read_file(
        roads_path,
        columns=[
            "highway",
            "maxspeed",
            "geometry",
        ],
    )

    if roads.empty:
        raise RuntimeError(
            "Road dataset is empty."
        )

    if roads.crs is None:
        raise RuntimeError(
            "Road dataset has no CRS."
        )

    print(
        f"Loaded road features: {len(roads):,}"
    )

    roads = roads[
        roads.geometry.notna()
    ]

    roads = roads[
        ~roads.geometry.is_empty
    ].copy()

    print(
        f"Valid road geometries: {len(roads):,}"
    )

    roads = roads.to_crs(
        output_crs
    )

    edge_records: list[dict[str, Any]] = []

    next_edge_id = 0

    total = len(
        roads
    )

    for start in range(
        0,
        total,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            total,
        )

        chunk = roads.iloc[
            start:end
        ]

        next_edge_id = process_chunk(
            chunk,
            edge_records,
            next_edge_id,
        )

        del chunk

        print(
            f"Processed roads: {end:,}/{total:,}"
        )

    if not edge_records:
        raise RuntimeError(
            "No road segments were generated."
        )

    edges = pd.DataFrame(
        edge_records
    )

    start_geometry = gpd.GeoSeries.from_xy(
        edges[
            "u_x"
        ],
        edges[
            "u_y"
        ],
    )

    end_geometry = gpd.GeoSeries.from_xy(
        edges[
            "v_x"
        ],
        edges[
            "v_y"
        ],
    )

    line_geometry = [
        LineString(
            [
                start,
                end,
            ]
        )
        for start, end in zip(
            start_geometry,
            end_geometry,
        )
    ]

    edges = gpd.GeoDataFrame(
        edges,
        geometry=line_geometry,
        crs=output_crs,
    )

    return edges


def build_compact_graph(
    edge_gdf: gpd.GeoDataFrame,
) -> nx.Graph:
    graph = nx.Graph()

    for row in edge_gdf.itertuples(
        index=False
    ):
        u = (
            int(
                round(
                    row.u_x
                    / COORDINATE_PRECISION
                )
            ),
            int(
                round(
                    row.u_y
                    / COORDINATE_PRECISION
                )
            ),
        )

        v = (
            int(
                round(
                    row.v_x
                    / COORDINATE_PRECISION
                )
            ),
            int(
                round(
                    row.v_y
                    / COORDINATE_PRECISION
                )
            ),
        )

        if u == v:
            continue

        graph.add_edge(
            u,
            v,
            edge_id=int(
                row.edge_id
            ),
            length_m=float(
                row.length_m
            ),
            travel_time_s=float(
                row.travel_time_s
            ),
            speed_kmh=float(
                row.speed_kmh
            ),
            flood_risk=0.0,
            bridge_risk=0.0,
            uncertainty_penalty=0.0,
            risk_cost=float(
                row.travel_time_s
            ),
        )

    graph.graph[
        "crs"
    ] = edge_gdf.crs.to_string()

    graph.graph[
        "source"
    ] = "OSM road network"

    return graph


def save_road_edges(
    edge_gdf: gpd.GeoDataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_gdf.to_file(
        output_path,
        driver="GPKG",
    )


def save_road_graph(
    graph: nx.Graph,
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
        "wb"
    ) as file:
        pickle.dump(
            graph,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def calculate_edge_metrics(
    edge_gdf: gpd.GeoDataFrame,
) -> dict:
    graph = build_compact_graph(
        edge_gdf
    )

    components = list(
        nx.connected_components(
            graph
        )
    )

    component_sizes = [
        len(component)
        for component in components
    ]

    total_length_km = float(
        edge_gdf[
            "length_m"
        ].sum()
        / 1000.0
    )

    return {
        "source": "OSM",
        "crs": str(
            edge_gdf.crs
        ),
        "road_segments": int(
            len(edge_gdf)
        ),
        "nodes": int(
            graph.number_of_nodes()
        ),
        "undirected_edges": int(
            graph.number_of_edges()
        ),
        "connected_components": int(
            len(components)
        ),
        "largest_component_nodes": int(
            max(
                component_sizes
            )
            if component_sizes
            else 0
        ),
        "total_road_length_km": total_length_km,
        "mean_segment_length_m": float(
            edge_gdf[
                "length_m"
            ].mean()
        ),
        "median_segment_length_m": float(
            edge_gdf[
                "length_m"
            ].median()
        ),
        "mean_speed_kmh": float(
            edge_gdf[
                "speed_kmh"
            ].mean()
        ),
        "flood_risk_initialized": False,
        "bridge_risk_initialized": False,
    }


def save_metrics(
    metrics: dict,
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
            metrics,
            file,
            indent=2,
        )
