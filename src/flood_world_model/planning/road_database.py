from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pyogrio


DEFAULT_SPEED_KMH = 30.0
CHUNK_SIZE = 5000
OUTPUT_CRS = "EPSG:32646"
COORDINATE_PRECISION_M = 1.0


def safe_float(
    value: Any,
    default: float,
) -> float:
    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def parse_speed(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(
        value,
        float,
    ) and math.isnan(value):
        return None

    if isinstance(
        value,
        str,
    ):
        text = value.strip()

        if not text:
            return None

        if ";" in text:
            text = text.split(
                ";"
            )[0]

        number = ""

        for character in text:
            if (
                character.isdigit()
                or character == "."
            ):
                number += character

        if number:
            parsed = safe_float(
                number,
                DEFAULT_SPEED_KMH,
            )

            if 5.0 <= parsed <= 150.0:
                return parsed

        return None

    parsed = safe_float(
        value,
        DEFAULT_SPEED_KMH,
    )

    if 5.0 <= parsed <= 150.0:
        return parsed

    return None


def estimate_speed_kmh(
    highway: str,
    maxspeed: Any,
) -> float:
    parsed = parse_speed(
        maxspeed
    )

    if parsed is not None:
        return parsed

    highway = str(
        highway or ""
    ).lower()

    speed_table = {
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
        "living_street": 15.0,
        "service": 15.0,
        "track": 10.0,
        "path": 5.0,
        "footway": 5.0,
        "pedestrian": 5.0,
    }

    return speed_table.get(
        highway,
        DEFAULT_SPEED_KMH,
    )


def coordinate_key(
    x: float,
    y: float,
) -> tuple[float, float]:
    return (
        round(
            float(x),
            1,
        ),
        round(
            float(y),
            1,
        ),
    )


def geometry_parts(
    geometry: Any,
):
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
            part
            for part in geometry.geoms
            if not part.is_empty
        ]

    return []


def initialize_database(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    connection.execute(
        "PRAGMA temp_store=MEMORY"
    )

    connection.execute(
        "PRAGMA cache_size=-65536"
    )

    connection.execute(
        "PRAGMA foreign_keys=ON"
    )

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER PRIMARY KEY,
            x REAL NOT NULL,
            y REAL NOT NULL,
            UNIQUE(x, y)
        );

        CREATE TABLE IF NOT EXISTS edges (
            edge_id INTEGER PRIMARY KEY,
            u INTEGER NOT NULL,
            v INTEGER NOT NULL,
            road_feature_id TEXT,
            highway TEXT,
            length_m REAL NOT NULL,
            speed_kmh REAL NOT NULL,
            travel_time_s REAL NOT NULL,
            FOREIGN KEY(u) REFERENCES nodes(node_id),
            FOREIGN KEY(v) REFERENCES nodes(node_id)
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_xy
        ON nodes(x, y);

        CREATE INDEX IF NOT EXISTS idx_edges_u
        ON edges(u);

        CREATE INDEX IF NOT EXISTS idx_edges_v
        ON edges(v);

        CREATE INDEX IF NOT EXISTS idx_edges_highway
        ON edges(highway);
        """
    )

    connection.commit()


def insert_nodes(
    connection: sqlite3.Connection,
    coordinates: set[tuple[float, float]],
) -> None:
    if not coordinates:
        return

    connection.executemany(
        """
        INSERT OR IGNORE INTO nodes (
            x,
            y
        )
        VALUES (?, ?)
        """,
        list(
            coordinates
        ),
    )


def get_node_ids(
    connection: sqlite3.Connection,
    coordinates: set[tuple[float, float]],
) -> dict[tuple[float, float], int]:
    result = {}

    for coordinate in coordinates:
        row = connection.execute(
            """
            SELECT node_id
            FROM nodes
            WHERE x = ?
            AND y = ?
            """,
            coordinate,
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Failed to resolve inserted road node."
            )

        result[
            coordinate
        ] = int(
            row[0]
        )

    return result


def process_chunk(
    connection: sqlite3.Connection,
    roads,
    next_edge_id: int,
) -> int:
    coordinates: set[
        tuple[float, float]
    ] = set()

    temporary_edges = []

    for feature_index, row in roads.iterrows():
        highway = str(
            row.get(
                "highway",
                "",
            )
        )

        speed_kmh = estimate_speed_kmh(
            highway,
            row.get(
                "maxspeed"
            ),
        )

        geometries = geometry_parts(
            row.geometry
        )

        for geometry in geometries:
            points = list(
                geometry.coords
            )

            if len(points) < 2:
                continue

            for index in range(
                len(points) - 1
            ):
                x1, y1 = points[
                    index
                ]

                x2, y2 = points[
                    index + 1
                ]

                u = coordinate_key(
                    x1,
                    y1,
                )

                v = coordinate_key(
                    x2,
                    y2,
                )

                if u == v:
                    continue

                ux, uy = u
                vx, vy = v

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

                coordinates.add(
                    u
                )

                coordinates.add(
                    v
                )

                temporary_edges.append(
                    (
                        u,
                        v,
                        str(
                            feature_index
                        ),
                        highway,
                        float(
                            length_m
                        ),
                        float(
                            speed_kmh
                        ),
                        float(
                            travel_time_s
                        ),
                    )
                )

    insert_nodes(
        connection,
        coordinates,
    )

    node_ids = get_node_ids(
        connection,
        coordinates,
    )

    edge_rows = []

    for (
        u,
        v,
        road_feature_id,
        highway,
        length_m,
        speed_kmh,
        travel_time_s,
    ) in temporary_edges:
        edge_rows.append(
            (
                next_edge_id,
                node_ids[u],
                node_ids[v],
                road_feature_id,
                highway,
                length_m,
                speed_kmh,
                travel_time_s,
            )
        )

        next_edge_id += 1

    if edge_rows:
        connection.executemany(
            """
            INSERT INTO edges (
                edge_id,
                u,
                v,
                road_feature_id,
                highway,
                length_m,
                speed_kmh,
                travel_time_s
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            edge_rows,
        )

    return next_edge_id


def build_road_database(
    roads_path: str | Path,
    database_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    output_crs: str = OUTPUT_CRS,
) -> dict:
    roads_path = Path(
        roads_path
    )

    database_path = Path(
        database_path
    )

    if not roads_path.exists():
        raise FileNotFoundError(
            f"Road source not found: {roads_path}"
        )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if database_path.exists():
        database_path.unlink()

    print(
        "Reading road metadata..."
    )

    info = pyogrio.read_info(
        roads_path
    )

    total_features = int(
        info["features"]
    )

    print(
        f"Total road features: {total_features:,}"
    )

    print(
        "Creating SQLite road database..."
    )

    connection = sqlite3.connect(
        database_path
    )

    initialize_database(
        connection
    )

    next_edge_id = 0
    processed_features = 0

    while (
        processed_features
        < total_features
    ):
        count = min(
            chunk_size,
            total_features
            - processed_features,
        )

        roads = pyogrio.read_dataframe(
            roads_path,
            columns=[
                "highway",
                "maxspeed",
                "geometry",
            ],
            skip_features=processed_features,
            max_features=count,
            use_arrow=False,
        )

        if roads.empty:
            break

        if roads.crs is None:
            connection.close()

            raise RuntimeError(
                "Road source has no CRS."
            )

        roads = roads.to_crs(
            output_crs
        )

        next_edge_id = process_chunk(
            connection,
            roads,
            next_edge_id,
        )

        connection.commit()

        processed_features += len(
            roads
        )

        print(
            f"Processed: {processed_features:,}/{total_features:,}"
        )

        del roads

    print(
        "Creating database statistics..."
    )

    connection.execute(
        "ANALYZE"
    )

    connection.commit()

    node_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM nodes
            """
        ).fetchone()[0]
    )

    edge_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM edges
            """
        ).fetchone()[0]
    )

    total_length_km = float(
        connection.execute(
            """
            SELECT COALESCE(
                SUM(length_m),
                0
            )
            FROM edges
            """
        ).fetchone()[0]
        / 1000.0
    )

    mean_segment_length = float(
        connection.execute(
            """
            SELECT COALESCE(
                AVG(length_m),
                0
            )
            FROM edges
            """
        ).fetchone()[0]
    )

    database_size_mb = (
        database_path.stat().st_size
        / 1024.0
        / 1024.0
    )

    connection.close()

    return {
        "source": "OSM",
        "source_path": str(
            roads_path
        ),
        "database": str(
            database_path
        ),
        "output_crs": output_crs,
        "coordinate_precision_m": COORDINATE_PRECISION_M,
        "chunk_size": chunk_size,
        "road_features": processed_features,
        "nodes": node_count,
        "edges": edge_count,
        "total_road_length_km": total_length_km,
        "mean_segment_length_m": mean_segment_length,
        "database_size_mb": database_size_mb,
        "networkx_global_graph": False,
    }


def inspect_road_database(
    database_path: str | Path,
) -> dict:
    database_path = Path(
        database_path
    )

    if not database_path.exists():
        raise FileNotFoundError(
            f"Road database not found: {database_path}"
        )

    connection = sqlite3.connect(
        database_path
    )

    nodes = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM nodes
            """
        ).fetchone()[0]
    )

    edges = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM edges
            """
        ).fetchone()[0]
    )

    length_km = float(
        connection.execute(
            """
            SELECT COALESCE(
                SUM(length_m),
                0
            )
            FROM edges
            """
        ).fetchone()[0]
        / 1000.0
    )

    connection.close()

    return {
        "database": str(
            database_path
        ),
        "nodes": nodes,
        "edges": edges,
        "total_road_length_km": length_km,
    }
