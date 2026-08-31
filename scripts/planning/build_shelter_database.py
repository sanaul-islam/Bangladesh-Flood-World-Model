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

from flood_world_model.planning.shelter import (
    build_shelter_database,
)


SHELTER_PATH = (
    PROJECT_ROOT
    / "data/static/shelters/shelters.shp"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/shelter_mapping.json"
)

OUTPUT_CRS = "EPSG:32646"

CHUNK_SIZE = 200

MAX_ROAD_DISTANCE_M = 5000.0


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("MEMORY-SAFE SHELTER DATABASE")
    print("=" * 80)

    print(
        f"Shelter source: {SHELTER_PATH}"
    )

    print(
        f"Road database: {DATABASE_PATH}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Maximum road snap distance: {MAX_ROAD_DISTANCE_M} m"
    )

    if not SHELTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing shelter dataset: {SHELTER_PATH}"
        )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing road database: {DATABASE_PATH}"
        )

    metrics = build_shelter_database(
        shelter_path=SHELTER_PATH,
        database_path=DATABASE_PATH,
        output_crs=OUTPUT_CRS,
        chunk_size=CHUNK_SIZE,
        max_road_distance_m=MAX_ROAD_DISTANCE_M,
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
    print("SHELTER DATABASE COMPLETE")
    print("=" * 80)

    print(
        f"Source features: {metrics['source_features']:,}"
    )

    print(
        f"Shelters stored: {metrics['shelters_stored']:,}"
    )

    print(
        f"Shelters mapped to roads: {metrics['shelters_mapped_to_roads']:,}"
    )

    print(
        f"Unmapped shelters: {metrics['shelters_unmapped']:,}"
    )

    print(
        f"Mapping rate: {metrics['mapping_rate'] * 100.0:.2f}%"
    )

    print(
        f"Average shelter-road distance: {metrics['average_shelter_road_distance_m']:.2f} m"
    )

    print(
        f"Maximum shelter-road distance: {metrics['maximum_shelter_road_distance_m']:.2f} m"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
