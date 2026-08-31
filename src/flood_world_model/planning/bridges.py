from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

import pyogrio


DEFAULT_GRID_SIZE_M = 1000.0
DEFAULT_MAX_BRIDGE_DISTANCE_M = 150.0
DEFAULT_CHUNK_SIZE = 2000


def safe_float(
    value: Any,
    default: float,
) -> float:
    try:
        result = float(value)

        if not math.isfinite(
            result
        ):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def initialize_bridge_tables(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS bridges (
            bridge_id INTEGER PRIMARY KEY,
            source_feature_id TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_bridges_xy
        ON bridges(x, y);

        CREATE TABLE IF NOT EXISTS bridge_node_map (
            bridge_id INTEGER PRIMARY KEY,
            node_id INTEGER NOT NULL,
            distance_m REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_bridge_node_map_node
        ON bridge_node_map(node_id);

        CREATE TABLE IF NOT EXISTS bridge_edge_map (
            bridge_id INTEGER NOT NULL,
            edge_id INTEGER NOT NULL,
            distance_m REAL NOT NULL,
            PRIMARY KEY (
                bridge_id,
                edge_id
            )
        );

        CREATE INDEX IF NOT EXISTS idx_bridge_edge_map_bridge
        ON bridge_edge_map(bridge_id);

        CREATE INDEX IF NOT EXISTS idx_bridge_edge_map_edge
        ON bridge_edge_map(edge_id);

        CREATE TABLE IF NOT EXISTS node_grid (
            gx INTEGER NOT NULL,
            gy INTEGER NOT NULL,
            node_id INTEGER NOT NULL,
            PRIMARY KEY (
                gx,
                gy,
                node_id
            )
        );

        CREATE INDEX IF NOT EXISTS idx_node_grid_gx_gy
        ON node_grid(gx, gy);
        """
    )

    connection.commit()


def node_grid_is_built(
    connection: sqlite3.Connection,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM node_grid
        LIMIT 1
        """
    ).fetchone()

    return row is not None


def build_node_grid(
    connection: sqlite3.Connection,
    grid_size_m: float,
) -> None:
    if node_grid_is_built(
        connection
    ):
        print(
            "Node spatial grid already exists."
        )

        return

    print(
        "Building disk-backed node spatial grid..."
    )

    connection.execute(
        "DELETE FROM node_grid"
    )

    connection.execute(
        """
        INSERT INTO node_grid (
            gx,
            gy,
            node_id
        )
        SELECT
            CAST(
                FLOOR(x / ?)
                AS INTEGER
            ),
            CAST(
                FLOOR(y / ?)
                AS INTEGER
            ),
            node_id
        FROM nodes
        """,
        (
            grid_size_m,
            grid_size_m,
        ),
    )

    connection.commit()

    count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM node_grid
            """
        ).fetchone()[0]
    )

    print(
        f"Node grid entries: {count:,}"
    )


def find_nearest_node(
    connection: sqlite3.Connection,
    x: float,
    y: float,
    grid_size_m: float,
    max_distance_m: float,
) -> tuple[int, float] | None:
    gx = math.floor(
        x / grid_size_m
    )

    gy = math.floor(
        y / grid_size_m
    )

    cell_radius = int(
        math.ceil(
            max_distance_m
            / grid_size_m
        )
    )

    min_gx = (
        gx - cell_radius
    )

    max_gx = (
        gx + cell_radius
    )

    min_gy = (
        gy - cell_radius
    )

    max_gy = (
        gy + cell_radius
    )

    rows = connection.execute(
        """
        SELECT
            n.node_id,
            n.x,
            n.y
        FROM node_grid AS g
        JOIN nodes AS n
            ON n.node_id = g.node_id
        WHERE
            g.gx BETWEEN ? AND ?
            AND g.gy BETWEEN ? AND ?
        """,
        (
            min_gx,
            max_gx,
            min_gy,
            max_gy,
        ),
    ).fetchall()

    if not rows:
        return None

    best_node_id = None
    best_distance = float(
        "inf"
    )

    max_distance_squared = (
        max_distance_m
        * max_distance_m
    )

    for (
        node_id,
        node_x,
        node_y,
    ) in rows:

        dx = (
            float(node_x)
            - x
        )

        dy = (
            float(node_y)
            - y
        )

        distance_squared = (
            dx * dx
            + dy * dy
        )

        if (
            distance_squared
            > max_distance_squared
        ):
            continue

        if (
            distance_squared
            < (
                best_distance
                * best_distance
            )
        ):
            best_distance = math.sqrt(
                distance_squared
            )

            best_node_id = int(
                node_id
            )

    if best_node_id is None:
        return None

    return (
        best_node_id,
        best_distance,
    )


def map_node_to_edges(
    connection: sqlite3.Connection,
    bridge_id: int,
    node_id: int,
    distance_m: float,
) -> int:
    rows = connection.execute(
        """
        SELECT edge_id
        FROM edges
        WHERE
            u = ?
            OR v = ?
        """,
        (
            node_id,
            node_id,
        ),
    ).fetchall()

    if not rows:
        return 0

    values = [
        (
            bridge_id,
            int(
                edge_id
            ),
            float(
                distance_m
            ),
        )
        for (
            edge_id,
        ) in rows
    ]

    connection.executemany(
        """
        INSERT OR REPLACE INTO bridge_edge_map (
            bridge_id,
            edge_id,
            distance_m
        )
        VALUES (?, ?, ?)
        """,
        values,
    )

    return len(
        values
    )


def process_bridge_chunk(
    connection: sqlite3.Connection,
    bridges,
    next_bridge_id: int,
    grid_size_m: float,
    max_distance_m: float,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    processed = 0
    mapped = 0
    edge_mappings = 0

    for feature_index, row in bridges.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        if geometry.is_empty:
            continue

        if geometry.geom_type == "Point":
            x = float(
                geometry.x
            )

            y = float(
                geometry.y
            )

        elif geometry.geom_type == "MultiPoint":
            points = [
                point
                for point in geometry.geoms
                if not point.is_empty
            ]

            if not points:
                continue

            x = float(
                np.mean(
                    [
                        point.x
                        for point in points
                    ]
                )
            )

            y = float(
                np.mean(
                    [
                        point.y
                        for point in points
                    ]
                )
            )

        else:
            # Some OSM-derived bridge layers can contain lines.
            # Use the representative point rather than loading a
            # separate spatial index.
            point = geometry.representative_point()

            x = float(
                point.x
            )

            y = float(
                point.y
            )

        bridge_id = (
            next_bridge_id
        )

        connection.execute(
            """
            INSERT INTO bridges (
                bridge_id,
                source_feature_id,
                x,
                y
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                bridge_id,
                str(
                    feature_index
                ),
                x,
                y,
            ),
        )

        processed += 1

        nearest = find_nearest_node(
            connection=connection,
            x=x,
            y=y,
            grid_size_m=grid_size_m,
            max_distance_m=max_distance_m,
        )

        if nearest is None:
            next_bridge_id += 1
            continue

        node_id, distance_m = nearest

        connection.execute(
            """
            INSERT OR REPLACE INTO bridge_node_map (
                bridge_id,
                node_id,
                distance_m
            )
            VALUES (?, ?, ?)
            """,
            (
                bridge_id,
                node_id,
                distance_m,
            ),
        )

        edge_count = map_node_to_edges(
            connection=connection,
            bridge_id=bridge_id,
            node_id=node_id,
            distance_m=distance_m,
        )

        if edge_count > 0:
            mapped += 1
            edge_mappings += edge_count

        next_bridge_id += 1

    return (
        next_bridge_id,
        processed,
        mapped,
        edge_mappings,
    )


def build_bridge_mapping(
    bridge_path: str | Path,
    database_path: str | Path,
    grid_size_m: float = DEFAULT_GRID_SIZE_M,
    max_distance_m: float = DEFAULT_MAX_BRIDGE_DISTANCE_M,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_crs: str = "EPSG:32646",
) -> dict:
    bridge_path = Path(
        bridge_path
    )

    database_path = Path(
        database_path
    )

    if not bridge_path.exists():
        raise FileNotFoundError(
            f"Bridge source not found: {bridge_path}"
        )

    if not database_path.exists():
        raise FileNotFoundError(
            f"Road database not found: {database_path}"
        )

    info = pyogrio.read_info(
        bridge_path
    )

    total_features = int(
        info["features"]
    )

    source_crs = info.get(
        "crs"
    )

    if source_crs is None:
        raise RuntimeError(
            "Bridge source has no CRS."
        )

    print(
        f"Bridge features: {total_features:,}"
    )

    print(
        f"Bridge source CRS: {source_crs}"
    )

    print(
        f"Road database: {database_path}"
    )

    print(
        f"Output CRS: {output_crs}"
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    connection.execute(
        "PRAGMA cache_size=-32768"
    )

    initialize_bridge_tables(
        connection
    )

    build_node_grid(
        connection,
        grid_size_m,
    )

    connection.execute(
        "DELETE FROM bridges"
    )

    connection.execute(
        "DELETE FROM bridge_node_map"
    )

    connection.execute(
        "DELETE FROM bridge_edge_map"
    )

    connection.commit()

    processed_features = 0
    next_bridge_id = 0
    processed_bridges = 0
    mapped_bridges = 0
    bridge_edge_mappings = 0

    while (
        processed_features
        < total_features
    ):
        count = min(
            chunk_size,
            total_features
            - processed_features,
        )

        bridges = pyogrio.read_dataframe(
            bridge_path,
            columns=[
                "geometry",
            ],
            skip_features=processed_features,
            max_features=count,
            use_arrow=False,
        )

        if bridges.empty:
            break

        bridges = bridges.to_crs(
            output_crs
        )

        (
            next_bridge_id,
            chunk_processed,
            chunk_mapped,
            chunk_edges,
        ) = process_bridge_chunk(
            connection=connection,
            bridges=bridges,
            next_bridge_id=next_bridge_id,
            grid_size_m=grid_size_m,
            max_distance_m=max_distance_m,
        )

        connection.commit()

        processed_features += len(
            bridges
        )

        processed_bridges += (
            chunk_processed
        )

        mapped_bridges += (
            chunk_mapped
        )

        bridge_edge_mappings += (
            chunk_edges
        )

        print(
            f"Processed bridges: {processed_features:,}/{total_features:,}"
        )

        print(
            f"Mapped bridges: {mapped_bridges:,}"
        )

        del bridges

    connection.execute(
        "ANALYZE"
    )

    connection.commit()

    actual_bridge_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM bridges
            """
        ).fetchone()[0]
    )

    mapped_node_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM bridge_node_map
            """
        ).fetchone()[0]
    )

    mapped_edge_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM bridge_edge_map
            """
        ).fetchone()[0]
    )

    average_distance_row = (
        connection.execute(
            """
            SELECT AVG(distance_m)
            FROM bridge_node_map
            """
        ).fetchone()
    )

    average_distance = float(
        average_distance_row[0]
        if average_distance_row
        and average_distance_row[0]
        is not None
        else 0.0
    )

    maximum_distance_row = (
        connection.execute(
            """
            SELECT MAX(distance_m)
            FROM bridge_node_map
            """
        ).fetchone()
    )

    maximum_distance = float(
        maximum_distance_row[0]
        if maximum_distance_row
        and maximum_distance_row[0]
        is not None
        else 0.0
    )

    database_size_mb = (
        database_path.stat().st_size
        / 1024.0
        / 1024.0
    )

    connection.close()

    mapping_rate = (
        mapped_bridges
        / max(
            processed_bridges,
            1,
        )
    )

    return {
        "bridge_source": str(
            bridge_path
        ),
        "road_database": str(
            database_path
        ),
        "source_crs": source_crs,
        "output_crs": output_crs,
        "total_source_features": total_features,
        "processed_bridge_features": processed_bridges,
        "mapped_bridges": mapped_bridges,
        "unmapped_bridges": (
            processed_bridges
            - mapped_bridges
        ),
        "mapping_rate": float(
            mapping_rate
        ),
        "bridge_node_mappings": mapped_node_count,
        "bridge_edge_mappings": mapped_edge_count,
        "average_bridge_node_distance_m": average_distance,
        "maximum_bridge_node_distance_m": maximum_distance,
        "grid_size_m": grid_size_m,
        "maximum_mapping_distance_m": max_distance_m,
        "chunk_size": chunk_size,
        "database_size_mb": database_size_mb,
        "networkx_global_graph": False,
    }
