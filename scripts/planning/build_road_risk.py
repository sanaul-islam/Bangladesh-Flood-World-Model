from __future__ import annotations

import json
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

from flood_world_model.planning.road_risk import (
    build_road_risk_snapshot,
)


DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

HAZARD_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_hydrological_hazard.nc"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/"
    "road_risk.json"
)

FORECAST_SAMPLE = 0

FORECAST_DAY = 1

BATCH_SIZE = 10000


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("FORECAST-AWARE ROAD RISK")
    print("=" * 80)

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Hazard: {HAZARD_PATH}"
    )

    print(
        f"Forecast sample: {FORECAST_SAMPLE}"
    )

    print(
        f"Forecast day: {FORECAST_DAY}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing database: {DATABASE_PATH}"
        )

    if not HAZARD_PATH.exists():
        raise FileNotFoundError(
            f"Missing hazard dataset: {HAZARD_PATH}"
        )

    metrics = build_road_risk_snapshot(
        database_path=DATABASE_PATH,
        hazard_path=HAZARD_PATH,
        forecast_sample=FORECAST_SAMPLE,
        forecast_day=FORECAST_DAY,
        batch_size=BATCH_SIZE,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("=" * 80)
    print("ROAD RISK COMPLETE")
    print("=" * 80)

    print(
        f"Processed edges: {metrics['processed_edges']:,}"
    )

    print(
        f"Risk records: {metrics['risk_records']:,}"
    )

    print(
        f"Mean flood risk: {metrics['mean_flood_risk']:.4f}"
    )

    print(
        f"Maximum flood risk: {metrics['max_flood_risk']:.4f}"
    )

    print(
        f"Mean uncertainty risk: {metrics['mean_uncertainty_risk']:.4f}"
    )

    print(
        f"Mean total risk: {metrics['mean_total_risk']:.4f}"
    )

    print(
        f"Maximum total risk: {metrics['max_total_risk']:.4f}"
    )

    print(
        f"Bridge-associated edges: {metrics['bridge_associated_edges']:,}"
    )

    print(
        f"Risk formula: {metrics['risk_formula']}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
