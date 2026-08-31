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

from flood_world_model.planning.bridges import (
    build_bridge_mapping,
)


BRIDGE_PATH = (
    PROJECT_ROOT
    / "data/static/bridges/bridges.shp"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/bridge_mapping.json"
)

GRID_SIZE_M = 1000.0

MAX_BRIDGE_DISTANCE_M = 150.0

CHUNK_SIZE = 2000

OUTPUT_CRS = "EPSG:32646"


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("MEMORY-SAFE BRIDGE-ROAD INTEGRATION")
    print("=" * 80)

    print(
        f"Bridge source: {BRIDGE_PATH}"
    )

    print(
        f"Road database: {DATABASE_PATH}"
    )

    print(
        f"Grid size: {GRID_SIZE_M} m"
    )

    print(
        f"Maximum bridge mapping distance: {MAX_BRIDGE_DISTANCE_M} m"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        "No nationwide NetworkX graph will be created."
    )

    if not BRIDGE_PATH.exists():
        raise FileNotFoundError(
            f"Missing bridge file: {BRIDGE_PATH}"
        )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing road database: {DATABASE_PATH}"
        )

    metrics = build_bridge_mapping(
        bridge_path=BRIDGE_PATH,
        database_path=DATABASE_PATH,
        grid_size_m=GRID_SIZE_M,
        max_distance_m=MAX_BRIDGE_DISTANCE_M,
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
    print("BRIDGE MAPPING COMPLETE")
    print("=" * 80)

    print(
        f"Source bridge features: {metrics['total_source_features']:,}"
    )

    print(
        f"Processed bridges: {metrics['processed_bridge_features']:,}"
    )

    print(
        f"Mapped bridges: {metrics['mapped_bridges']:,}"
    )

    print(
        f"Unmapped bridges: {metrics['unmapped_bridges']:,}"
    )

    print(
        f"Mapping rate: {metrics['mapping_rate'] * 100.0:.2f}%"
    )

    print(
        f"Bridge-node mappings: {metrics['bridge_node_mappings']:,}"
    )

    print(
        f"Bridge-edge mappings: {metrics['bridge_edge_mappings']:,}"
    )

    print(
        f"Average bridge-node distance: {metrics['average_bridge_node_distance_m']:.2f} m"
    )

    print(
        f"Maximum bridge-node distance: {metrics['maximum_bridge_node_distance_m']:.2f} m"
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
