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

from flood_world_model.planning.road_database import (
    build_road_database,
)


ROADS_PATH = (
    PROJECT_ROOT
    / "data/static/roads/roads.shp"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/road_network.json"
)

CHUNK_SIZE = 5000

OUTPUT_CRS = "EPSG:32646"


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("MEMORY-SAFE DISK-BACKED ROAD NETWORK")
    print("=" * 80)

    print(
        f"Source: {ROADS_PATH}"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        "Important: no nationwide NetworkX graph will be created."
    )

    if not ROADS_PATH.exists():
        raise FileNotFoundError(
            f"Missing roads file: {ROADS_PATH}"
        )

    print(
        "Building SQLite road database..."
    )

    metrics = build_road_database(
        roads_path=ROADS_PATH,
        database_path=DATABASE_PATH,
        chunk_size=CHUNK_SIZE,
        output_crs=OUTPUT_CRS,
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
    print("ROAD DATABASE COMPLETE")
    print("=" * 80)

    print(
        f"Road features: {metrics['road_features']:,}"
    )

    print(
        f"Nodes: {metrics['nodes']:,}"
    )

    print(
        f"Edges: {metrics['edges']:,}"
    )

    print(
        f"Total road length: {metrics['total_road_length_km']:.2f} km"
    )

    print(
        f"Mean segment length: {metrics['mean_segment_length_m']:.2f} m"
    )

    if "median_segment_length_m" in metrics:
        print(
            f"Median segment length: {metrics['median_segment_length_m']:.2f} m"
        )

    if "mean_speed_kmh" in metrics:
        print(
            f"Mean speed: {metrics['mean_speed_kmh']:.2f} km/h"
        )

    print(
        f"Database size: {metrics['database_size_mb']:.2f} MB"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
