from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pyogrio


DEFAULT_MAX_SHELTER_ROAD_DISTANCE_M = 5000.0
DEFAULT_CHUNK_SIZE = 500


def initialize_shelter_tables(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS shelters (
            shelter_id INTEGER PRIMARY KEY,
            source_feature_id TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_shelters_xy
        ON shelters(x, y);

        CREATE TABLE IF NOT EXISTS shelter_node_map (
            shelter_id INTEGER PRIMARY KEY,
            node_id INTEGER NOT NULL,
            distance_m REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_shelter_node_map_node
        ON shelter_node_map(node_id);

        CREATE TABLE IF NOT EXISTS shelter_scores (
            shelter_id INTEGER PRIMARY KEY,
            forecast_sample INTEGER NOT NULL,
            forecast_day INTEGER NOT NULL,
            hazard_score REAL NOT NULL,
            population_exposure REAL NOT NULL,
            road_distance_km REAL,
            estimated_route_time_min REAL,
            route_risk_cost REAL,
            bridge_edges INTEGER,
            accessibility_score REAL NOT NULL,
            combined_score REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_shelter_scores_forecast
        ON shelter_scores(
            forecast_sample,
            forecast_day
        );
        """
    )

    connection.commit()


def build_node_grid(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS node_grid (
            gx INTEGER NOT NULL,
            gy INTEGER NOT NULL,
            node_id INTEGER NOT NULL,
            PRIMARY KEY(
                gx,
                gy,
                node_id
            )
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_node_grid_cell
        ON node_grid(gx, gy)
        """
    )

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM node_grid
        """
    ).fetchone()[0]

    if count > 0:
        return

    connection.execute(
        """
        INSERT INTO node_grid(
            gx,
            gy,
            node_id
        )
        SELECT
            CAST(FLOOR(x / 1000.0) AS INTEGER),
            CAST(FLOOR(y / 1000.0) AS INTEGER),
            node_id
        FROM nodes
        """
    )

    connection.commit()


def nearest_node(
    connection: sqlite3.Connection,
    x: float,
    y: float,
    max_distance_m: float,
) -> tuple[int, float] | None:
    grid_x = math.floor(
        x / 1000.0
    )

    grid_y = math.floor(
        y / 1000.0
    )

    radius_cells = max(
        1,
        int(
            math.ceil(
                max_distance_m
                / 1000.0
            )
        ),
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
            grid_x - radius_cells,
            grid_x + radius_cells,
            grid_y - radius_cells,
            grid_y + radius_cells,
        ),
    ).fetchall()

    if not rows:
        return None

    best_node = None
    best_distance = float(
        "inf"
    )

    max_distance_squared = (
        max_distance_m
        * max_distance_m
    )

    for node_id, node_x, node_y in rows:
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
            <= max_distance_squared
            and distance_squared
            < best_distance * best_distance
        ):
            best_node = int(
                node_id
            )

            best_distance = math.sqrt(
                distance_squared
            )

    if best_node is None:
        return None

    return (
        best_node,
        best_distance,
    )


def build_shelter_database(
    shelter_path: str | Path,
    database_path: str | Path,
    output_crs: str = "EPSG:32646",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_road_distance_m: float = (
        DEFAULT_MAX_SHELTER_ROAD_DISTANCE_M
    ),
) -> dict:
    shelter_path = Path(
        shelter_path
    )

    database_path = Path(
        database_path
    )

    if not shelter_path.exists():
        raise FileNotFoundError(
            f"Shelter dataset not found: {shelter_path}"
        )

    if not database_path.exists():
        raise FileNotFoundError(
            f"Road database not found: {database_path}"
        )

    info = pyogrio.read_info(
        shelter_path
    )

    total_features = int(
        info["features"]
    )

    source_crs = info.get(
        "crs"
    )

    if source_crs is None:
        raise RuntimeError(
            "Shelter dataset has no CRS."
        )

    print(
        f"Shelter features: {total_features:,}"
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    connection.execute(
        "PRAGMA cache_size=-32768"
    )

    initialize_shelter_tables(
        connection
    )

    build_node_grid(
        connection
    )

    connection.execute(
        "DELETE FROM shelters"
    )

    connection.execute(
        "DELETE FROM shelter_node_map"
    )

    connection.commit()

    processed = 0
    shelter_id = 0
    mapped = 0

    while processed < total_features:
        count = min(
            chunk_size,
            total_features - processed,
        )

        shelters = pyogrio.read_dataframe(
            shelter_path,
            columns=[
                "geometry",
            ],
            skip_features=processed,
            max_features=count,
            use_arrow=False,
        )

        if shelters.empty:
            break

        shelters = shelters.to_crs(
            output_crs
        )

        for feature_index, row in shelters.iterrows():

            geometry = row.geometry

            if geometry is None:
                continue

            if geometry.is_empty:
                continue

            if geometry.geom_type == "Point":
                point = geometry

            else:
                point = geometry.representative_point()

            x = float(
                point.x
            )

            y = float(
                point.y
            )

            connection.execute(
                """
                INSERT INTO shelters(
                    shelter_id,
                    source_feature_id,
                    x,
                    y
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    shelter_id,
                    str(
                        feature_index
                    ),
                    x,
                    y,
                ),
            )

            nearest = nearest_node(
                connection=connection,
                x=x,
                y=y,
                max_distance_m=max_road_distance_m,
            )

            if nearest is not None:
                node_id, distance_m = nearest

                connection.execute(
                    """
                    INSERT INTO shelter_node_map(
                        shelter_id,
                        node_id,
                        distance_m
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        shelter_id,
                        node_id,
                        distance_m,
                    ),
                )

                mapped += 1

            shelter_id += 1

        connection.commit()

        processed += len(
            shelters
        )

        print(
            f"Processed shelters: {processed:,}/{total_features:,}"
        )

        print(
            f"Mapped shelters: {mapped:,}"
        )

        del shelters

    connection.execute(
        "ANALYZE"
    )

    connection.commit()

    total_shelters = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM shelters
            """
        ).fetchone()[0]
    )

    mapped_shelters = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM shelter_node_map
            """
        ).fetchone()[0]
    )

    average_distance = connection.execute(
        """
        SELECT AVG(distance_m)
        FROM shelter_node_map
        """
    ).fetchone()[0]

    maximum_distance = connection.execute(
        """
        SELECT MAX(distance_m)
        FROM shelter_node_map
        """
    ).fetchone()[0]

    connection.close()

    return {
        "source": "OSM",
        "source_path": str(
            shelter_path
        ),
        "database": str(
            database_path
        ),
        "output_crs": output_crs,
        "source_features": total_features,
        "shelters_stored": total_shelters,
        "shelters_mapped_to_roads": mapped_shelters,
        "shelters_unmapped": (
            total_shelters
            - mapped_shelters
        ),
        "mapping_rate": float(
            mapped_shelters
            / max(
                total_shelters,
                1,
            )
        ),
        "average_shelter_road_distance_m": float(
            average_distance
            or 0.0
        ),
        "maximum_shelter_road_distance_m": float(
            maximum_distance
            or 0.0
        ),
    }


def rank_shelters(
    database_path: str | Path,
    forecast_sample: int,
    forecast_day: int,
    origin_node: int,
    limit: int = 10,
) -> list[dict]:
    database_path = Path(
        database_path
    )

    connection = sqlite3.connect(
        database_path
    )

    rows = connection.execute(
        """
        SELECT
            shelter_id,
            x,
            y,
            hazard_score,
            population_exposure,
            road_distance_km,
            estimated_route_time_min,
            route_risk_cost,
            bridge_edges,
            accessibility_score,
            combined_score
        FROM shelter_scores
        WHERE
            forecast_sample = ?
            AND forecast_day = ?
        ORDER BY combined_score ASC
        LIMIT ?
        """,
        (
            forecast_sample,
            forecast_day,
            limit,
        ),
    ).fetchall()

    connection.close()

    results = []

    for row in rows:
        results.append(
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
                "hazard_score": float(
                    row[3]
                ),
                "population_exposure": float(
                    row[4]
                ),
                "road_distance_km": (
                    float(row[5])
                    if row[5] is not None
                    else None
                ),
                "estimated_route_time_min": (
                    float(row[6])
                    if row[6] is not None
                    else None
                ),
                "route_risk_cost": (
                    float(row[7])
                    if row[7] is not None
                    else None
                ),
                "bridge_edges": int(
                    row[8]
                ),
                "accessibility_score": float(
                    row[9]
                ),
                "combined_score": float(
                    row[10]
                ),
            }
        )

    return results
