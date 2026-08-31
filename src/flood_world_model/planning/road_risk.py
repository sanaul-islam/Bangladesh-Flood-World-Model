from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer


DEFAULT_BATCH_SIZE = 2000

FLOOD_WEIGHT = 2.0
BRIDGE_WEIGHT = 2.0
UNCERTAINTY_WEIGHT = 1.0


def initialize_road_risk_tables(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS road_risk;
        DROP TABLE IF EXISTS road_edge_state;

        CREATE TABLE road_risk (
            edge_id INTEGER PRIMARY KEY,
            forecast_sample INTEGER NOT NULL,
            forecast_day INTEGER NOT NULL,
            grid_lat_index INTEGER,
            grid_lon_index INTEGER,
            grid_distance_degrees REAL,
            flood_risk REAL NOT NULL,
            uncertainty_risk REAL NOT NULL,
            bridge_exposure REAL NOT NULL,
            bridge_risk REAL NOT NULL,
            total_risk REAL NOT NULL,
            risk_cost REAL NOT NULL
        );

        CREATE INDEX idx_road_risk_sample_day
        ON road_risk(
            forecast_sample,
            forecast_day
        );

        CREATE INDEX idx_road_risk_edge
        ON road_risk(
            edge_id
        );

        CREATE TABLE road_edge_state (
            edge_id INTEGER PRIMARY KEY,
            flood_risk REAL NOT NULL DEFAULT 0,
            uncertainty_risk REAL NOT NULL DEFAULT 0,
            bridge_exposure REAL NOT NULL DEFAULT 0,
            bridge_risk REAL NOT NULL DEFAULT 0,
            total_risk REAL NOT NULL DEFAULT 0,
            risk_cost REAL NOT NULL DEFAULT 0
        );
        """
    )

    connection.commit()


def load_valid_hazard_grid(
    hazard_path: str | Path,
    forecast_sample: int,
    forecast_day: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ds = xr.open_dataset(
        hazard_path
    )

    samples = ds[
        "sample"
    ].values

    forecast_days = ds[
        "forecast_day"
    ].values

    sample_matches = np.where(
        samples == forecast_sample
    )[0]

    day_matches = np.where(
        forecast_days == forecast_day
    )[0]

    if len(sample_matches) == 0:
        ds.close()

        raise ValueError(
            f"Forecast sample {forecast_sample} not found."
        )

    if len(day_matches) == 0:
        ds.close()

        raise ValueError(
            f"Forecast day {forecast_day} not found."
        )

    sample_position = int(
        sample_matches[0]
    )

    day_position = int(
        day_matches[0]
    )

    hazard = (
        ds[
            "hydrological_hazard_score"
        ]
        .isel(
            sample=sample_position,
            forecast_day=day_position,
        )
        .values
        .astype(np.float32)
    )

    uncertainty = (
        ds[
            "uncertainty_component"
        ]
        .isel(
            sample=sample_position,
            forecast_day=day_position,
        )
        .values
        .astype(np.float32)
    )

    lat = (
        ds[
            "lat"
        ]
        .values
        .astype(np.float64)
    )

    lon = (
        ds[
            "lon"
        ]
        .values
        .astype(np.float64)
    )

    ds.close()

    valid = (
        np.isfinite(
            hazard
        )
        & np.isfinite(
            uncertainty
        )
    )

    valid_count = int(
        valid.sum()
    )

    total_count = int(
        valid.size
    )

    print(
        f"Valid hazard cells: {valid_count:,}/{total_count:,}"
    )

    if valid_count < 10:
        raise RuntimeError(
            "Too few valid hazard cells."
        )

    rows, cols = np.where(
        valid
    )

    grid_points = np.column_stack(
        [
            lat[rows],
            lon[cols],
        ]
    ).astype(
        np.float64
    )

    return (
        hazard,
        uncertainty,
        rows.astype(np.int32),
        cols.astype(np.int32),
        grid_points,
    )


def nearest_grid_cells(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    grid_points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if grid_points.size == 0:
        raise RuntimeError(
            "No valid hazard grid points."
        )

    mean_lat = float(
        np.mean(
            grid_points[
                :,
                0,
            ]
        )
    )

    cos_lat = math.cos(
        math.radians(
            mean_lat
        )
    )

    query_lat = np.asarray(
        latitudes,
        dtype=np.float64,
    )

    query_lon = (
        np.asarray(
            longitudes,
            dtype=np.float64,
        )
        * cos_lat
    )

    grid_lat = (
        grid_points[
            :,
            0,
        ]
    )

    grid_lon = (
        grid_points[
            :,
            1,
        ]
        * cos_lat
    )

    nearest_positions = np.empty(
        len(query_lat),
        dtype=np.int32,
    )

    distances = np.empty(
        len(query_lat),
        dtype=np.float32,
    )

    block_size = 256

    for start in range(
        0,
        len(query_lat),
        block_size,
    ):
        end = min(
            start + block_size,
            len(query_lat),
        )

        qlat = query_lat[
            start:end
        ]

        qlon = query_lon[
            start:end
        ]

        lat_difference = (
            qlat[:, None]
            - grid_lat[None, :]
        )

        lon_difference = (
            qlon[:, None]
            - grid_lon[None, :]
        )

        distance_squared = (
            lat_difference ** 2
            + lon_difference ** 2
        )

        nearest = np.argmin(
            distance_squared,
            axis=1,
        )

        distance = np.sqrt(
            distance_squared[
                np.arange(
                    len(
                        nearest
                    )
                ),
                nearest,
            ]
        )

        nearest_positions[
            start:end
        ] = nearest.astype(
            np.int32
        )

        distances[
            start:end
        ] = distance.astype(
            np.float32
        )

        del (
            lat_difference,
            lon_difference,
            distance_squared,
        )

    return (
        nearest_positions,
        distances,
    )


def load_bridge_edge_ids(
    connection: sqlite3.Connection,
) -> set[int]:
    rows = connection.execute(
        """
        SELECT DISTINCT edge_id
        FROM bridge_edge_map
        """
    ).fetchall()

    return {
        int(
            row[0]
        )
        for row in rows
    }


def calculate_risk_cost(
    travel_time_s: float,
    flood_risk: float,
    bridge_risk: float,
    uncertainty_risk: float,
) -> float:
    return float(
        travel_time_s
        * (
            1.0
            + FLOOD_WEIGHT
            * flood_risk
            + BRIDGE_WEIGHT
            * bridge_risk
            + UNCERTAINTY_WEIGHT
            * uncertainty_risk
        )
    )


def build_road_risk_snapshot(
    database_path: str | Path,
    hazard_path: str | Path,
    forecast_sample: int,
    forecast_day: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    database_path = Path(
        database_path
    )

    hazard_path = Path(
        hazard_path
    )

    if not database_path.exists():
        raise FileNotFoundError(
            f"Road database not found: {database_path}"
        )

    if not hazard_path.exists():
        raise FileNotFoundError(
            f"Hazard dataset not found: {hazard_path}"
        )

    print(
        "Loading hazard grid..."
    )

    (
        hazard,
        uncertainty,
        valid_rows,
        valid_cols,
        grid_points,
    ) = load_valid_hazard_grid(
        hazard_path=hazard_path,
        forecast_sample=forecast_sample,
        forecast_day=forecast_day,
    )

    print(
        f"Valid spatial grid points: {len(grid_points):,}"
    )

    print(
        f"Hazard min: {float(np.nanmin(hazard)):.6f}"
    )

    print(
        f"Hazard max: {float(np.nanmax(hazard)):.6f}"
    )

    print(
        f"Hazard mean: {float(np.nanmean(hazard)):.6f}"
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

    initialize_road_risk_tables(
        connection
    )

    bridge_edge_ids = (
        load_bridge_edge_ids(
            connection
        )
    )

    print(
        f"Bridge-associated edges: {len(bridge_edge_ids):,}"
    )

    total_edges = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM edges
            """
        ).fetchone()[0]
    )

    print(
        f"Total road edges: {total_edges:,}"
    )

    transformer = Transformer.from_crs(
        "EPSG:32646",
        "EPSG:4326",
        always_xy=True,
    )

    cursor = connection.execute(
        """
        SELECT
            edge_id,
            u,
            v,
            length_m,
            speed_kmh,
            travel_time_s
        FROM edges
        ORDER BY edge_id
        """
    )

    processed = 0
    mapped = 0
    bridge_count = 0

    flood_sum = 0.0
    uncertainty_sum = 0.0
    total_risk_sum = 0.0

    max_flood = -np.inf
    max_uncertainty = -np.inf
    max_total_risk = -np.inf

    while True:
        rows = cursor.fetchmany(
            batch_size
        )

        if not rows:
            break

        count = len(
            rows
        )

        edge_ids = np.fromiter(
            (
                int(
                    row[0]
                )
                for row in rows
            ),
            dtype=np.int64,
            count=count,
        )

        u_x = np.fromiter(
            (
                float(
                    row[1]
                )
                for row in rows
            ),
            dtype=np.float64,
            count=count,
        )

        u_y = np.fromiter(
            (
                float(
                    row[2]
                )
                for row in rows
            ),
            dtype=np.float64,
            count=count,
        )

        v_x = np.fromiter(
            (
                float(
                    row[3]
                )
                for row in rows
            ),
            dtype=np.float64,
            count=count,
        )

        v_y = np.fromiter(
            (
                float(
                    row[4]
                )
                for row in rows
            ),
            dtype=np.float64,
            count=count,
        )

        travel_times = np.fromiter(
            (
                float(
                    row[5]
                )
                for row in rows
            ),
            dtype=np.float64,
            count=count,
        )

        midpoint_x = (
            u_x
            + v_x
        ) * 0.5

        midpoint_y = (
            u_y
            + v_y
        ) * 0.5

        longitudes, latitudes = (
            transformer.transform(
                midpoint_x,
                midpoint_y,
            )
        )

        (
            nearest_positions,
            grid_distances,
        ) = nearest_grid_cells(
            latitudes=np.asarray(
                latitudes,
                dtype=np.float64,
            ),
            longitudes=np.asarray(
                longitudes,
                dtype=np.float64,
            ),
            grid_points=grid_points,
        )

        records = []
        state_records = []

        for i in range(
            count
        ):
            edge_id = int(
                edge_ids[i]
            )

            position = int(
                nearest_positions[i]
            )

            grid_row = int(
                valid_rows[
                    position
                ]
            )

            grid_col = int(
                valid_cols[
                    position
                ]
            )

            flood_risk = float(
                hazard[
                    grid_row,
                    grid_col,
                ]
            )

            uncertainty_risk = float(
                uncertainty[
                    grid_row,
                    grid_col,
                ]
            )

            if not math.isfinite(
                flood_risk
            ):
                flood_risk = 0.0

            if not math.isfinite(
                uncertainty_risk
            ):
                uncertainty_risk = 0.0

            flood_risk = float(
                np.clip(
                    flood_risk,
                    0.0,
                    1.0,
                )
            )

            uncertainty_risk = float(
                np.clip(
                    uncertainty_risk,
                    0.0,
                    1.0,
                )
            )

            has_bridge = (
                edge_id
                in bridge_edge_ids
            )

            bridge_exposure = (
                1.0
                if has_bridge
                else 0.0
            )

            bridge_risk = (
                flood_risk
                * bridge_exposure
            )

            total_risk = float(
                np.clip(
                    (
                        FLOOD_WEIGHT
                        * flood_risk
                        + BRIDGE_WEIGHT
                        * bridge_risk
                        + UNCERTAINTY_WEIGHT
                        * uncertainty_risk
                    )
                    / (
                        FLOOD_WEIGHT
                        + BRIDGE_WEIGHT
                        + UNCERTAINTY_WEIGHT
                    ),
                    0.0,
                    1.0,
                )
            )

            risk_cost = calculate_risk_cost(
                travel_time_s=float(
                    travel_times[i]
                ),
                flood_risk=flood_risk,
                bridge_risk=bridge_risk,
                uncertainty_risk=uncertainty_risk,
            )

            records.append(
                (
                    edge_id,
                    forecast_sample,
                    forecast_day,
                    grid_row,
                    grid_col,
                    float(
                        grid_distances[i]
                    ),
                    flood_risk,
                    uncertainty_risk,
                    bridge_exposure,
                    bridge_risk,
                    total_risk,
                    risk_cost,
                )
            )

            state_records.append(
                (
                    edge_id,
                    flood_risk,
                    uncertainty_risk,
                    bridge_exposure,
                    bridge_risk,
                    total_risk,
                    risk_cost,
                )
            )

            mapped += 1

            flood_sum += flood_risk
            uncertainty_sum += uncertainty_risk
            total_risk_sum += total_risk

            max_flood = max(
                max_flood,
                flood_risk,
            )

            max_uncertainty = max(
                max_uncertainty,
                uncertainty_risk,
            )

            max_total_risk = max(
                max_total_risk,
                total_risk,
            )

            if has_bridge:
                bridge_count += 1

        connection.executemany(
            """
            INSERT INTO road_risk (
                edge_id,
                forecast_sample,
                forecast_day,
                grid_lat_index,
                grid_lon_index,
                grid_distance_degrees,
                flood_risk,
                uncertainty_risk,
                bridge_exposure,
                bridge_risk,
                total_risk,
                risk_cost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        connection.executemany(
            """
            INSERT OR REPLACE INTO road_edge_state (
                edge_id,
                flood_risk,
                uncertainty_risk,
                bridge_exposure,
                bridge_risk,
                total_risk,
                risk_cost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            state_records,
        )

        connection.commit()

        processed += count

        print(
            f"Processed: {processed:,}/{total_edges:,}"
        )

        del (
            rows,
            edge_ids,
            u_x,
            u_y,
            v_x,
            v_y,
            travel_times,
            midpoint_x,
            midpoint_y,
            longitudes,
            latitudes,
            nearest_positions,
            grid_distances,
            records,
            state_records,
        )

    connection.execute(
        "ANALYZE"
    )

    connection.commit()

    risk_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM road_risk
            WHERE forecast_sample = ?
            AND forecast_day = ?
            """,
            (
                forecast_sample,
                forecast_day,
            ),
        ).fetchone()[0]
    )

    connection.close()

    if risk_count == 0:
        raise RuntimeError(
            "No road risk records were created."
        )

    if mapped == 0:
        raise RuntimeError(
            "No road edges were mapped."
        )

    if (
        not np.isfinite(
            max_flood
        )
        or max_flood <= 0.0
    ):
        raise RuntimeError(
            "Maximum flood risk is zero. "
            "Hazard-to-road mapping failed."
        )

    mapping_rate = (
        mapped
        / max(
            processed,
            1,
        )
    )

    return {
        "forecast_sample": int(
            forecast_sample
        ),
        "forecast_day": int(
            forecast_day
        ),
        "processed_edges": int(
            processed
        ),
        "mapped_edges": int(
            mapped
        ),
        "mapping_rate": float(
            mapping_rate
        ),
        "risk_records": int(
            risk_count
        ),
        "mean_flood_risk": float(
            flood_sum
            / mapped
        ),
        "max_flood_risk": float(
            max_flood
        ),
        "mean_uncertainty_risk": float(
            uncertainty_sum
            / mapped
        ),
        "max_uncertainty_risk": float(
            max_uncertainty
        ),
        "mean_total_risk": float(
            total_risk_sum
            / mapped
        ),
        "max_total_risk": float(
            max_total_risk
        ),
        "bridge_associated_edges": int(
            bridge_count
        ),
        "risk_formula": (
            "travel_time × "
            "(1 + 2×flood_risk + "
            "2×bridge_risk + "
            "1×uncertainty_risk)"
        ),
    }
