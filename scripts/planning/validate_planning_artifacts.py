from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import xarray as xr


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

HAZARD_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_hydrological_hazard.nc"
)

POPULATION_RISK_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_population_risk.nc"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs/metrics"
)


def fail(message: str) -> None:
    raise RuntimeError(
        message
    )


def validate_database() -> dict:
    if not DATABASE_PATH.exists():
        fail(
            f"Missing database: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    required_tables = [
        "nodes",
        "edges",
        "bridges",
        "bridge_node_map",
        "bridge_edge_map",
        "road_risk",
        "road_edge_state",
        "shelters",
        "shelter_node_map",
    ]

    existing = {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }

    missing = [
        table
        for table in required_tables
        if table not in existing
    ]

    if missing:
        connection.close()

        fail(
            f"Missing database tables: {missing}"
        )

    counts = {}

    for table in required_tables:
        counts[
            table
        ] = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        )

    if counts["nodes"] < 1_000_000:
        connection.close()

        fail(
            "Unexpectedly small road-node database."
        )

    if counts["edges"] < 1_000_000:
        connection.close()

        fail(
            "Unexpectedly small road-edge database."
        )

    if counts["bridges"] < 1:
        connection.close()

        fail(
            "No bridges found."
        )

    if counts["shelters"] < 1:
        connection.close()

        fail(
            "No shelters found."
        )

    if counts["shelter_node_map"] < 1:
        connection.close()

        fail(
            "No shelters are mapped to roads."
        )

    road_risk_nonzero = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM road_risk
            WHERE flood_risk > 0
            """
        ).fetchone()[0]
    )

    road_risk_total = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM road_risk
            """
        ).fetchone()[0]
    )

    if road_risk_total == 0:
        connection.close()

        fail(
            "road_risk is empty."
        )

    if road_risk_nonzero == 0:
        connection.close()

        fail(
            "All road flood-risk values are zero."
        )

    road_state_total = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM road_edge_state
            """
        ).fetchone()[0]
    )

    if road_state_total == 0:
        connection.close()

        fail(
            "road_edge_state is empty."
        )

    connection.close()

    return {
        "counts": counts,
        "road_risk_records": road_risk_total,
        "nonzero_road_risk_records": road_risk_nonzero,
    }


def validate_forecasts() -> dict:
    for path in [
        HAZARD_PATH,
        POPULATION_RISK_PATH,
    ]:
        if not path.exists():
            fail(
                f"Missing forecast artifact: {path}"
            )

    with xr.open_dataset(
        HAZARD_PATH
    ) as ds:

        if "hydrological_hazard_score" not in ds:
            fail(
                "Hydrological hazard variable missing."
            )

        hazard_shape = ds[
            "hydrological_hazard_score"
        ].shape

        if hazard_shape != (
            606,
            7,
            60,
            45,
        ):
            fail(
                f"Unexpected hazard shape: {hazard_shape}"
            )

    with xr.open_dataset(
        POPULATION_RISK_PATH
    ) as ds:

        required = [
            "hydrological_hazard_score",
            "population_exposure_index",
            "lat",
            "lon",
            "sample",
            "forecast_day",
        ]

        missing = [
            name
            for name in required
            if name not in ds
        ]

        if missing:
            fail(
                f"Population-risk variables missing: {missing}"
            )

        exposure = ds[
            "population_exposure_index"
        ].isel(
            sample=0,
            forecast_day=1,
        ).values

        finite = (
            exposure[
                ~__import__(
                    "numpy"
                ).isnan(
                    exposure
                )
            ]
        )

        if finite.size == 0:
            fail(
                "Population exposure contains no finite values."
            )

        if float(
            finite.max()
        ) <= 0.0:
            fail(
                "Population exposure is entirely zero."
            )

    return {
        "hazard_shape": list(
            hazard_shape
        ),
        "population_risk": "valid",
    }


def main() -> None:
    print("=" * 80)
    print(
        "BANGLADESH FLOOD WORLD MODEL"
    )
    print(
        "PLANNING ARTIFACT VALIDATION"
    )
    print("=" * 80)

    database = validate_database()

    forecasts = validate_forecasts()

    report = {
        "database": database,
        "forecasts": forecasts,
    }

    output_path = (
        METRICS_DIR
        / "planning_artifact_validation.json"
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
            report,
            file,
            indent=2,
        )

    print("=" * 80)
    print("VALIDATION PASSED")
    print("=" * 80)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
