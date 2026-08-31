from __future__ import annotations

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

from flood_world_model.planning.graph import (
    build_compact_graph,
    build_road_edges,
    calculate_edge_metrics,
    save_metrics,
    save_road_edges,
    save_road_graph,
)


ROADS_PATH = (
    PROJECT_ROOT
    / "data/static/roads/roads.shp"
)

EDGE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_edges.gpkg"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data/processed/road_graph.gpickle"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/road_graph.json"
)

OUTPUT_CRS = "EPSG:32646"
CHUNK_SIZE = 10000


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("MEMORY-OPTIMIZED OSM ROAD GRAPH")
    print("=" * 80)

    if not ROADS_PATH.exists():
        raise FileNotFoundError(
            f"Missing roads file: {ROADS_PATH}"
        )

    print(
        f"Road source: {ROADS_PATH}"
    )

    print(
        "Building compact road edge table..."
    )

    edge_gdf = build_road_edges(
        roads_path=ROADS_PATH,
        output_crs=OUTPUT_CRS,
        chunk_size=CHUNK_SIZE,
    )

    print(
        f"Generated road segments: {len(edge_gdf):,}"
    )

    print(
        "Saving road edge GeoPackage..."
    )

    save_road_edges(
        edge_gdf=edge_gdf,
        output_path=EDGE_PATH,
    )

    print(
        "Building NetworkX graph..."
    )

    graph = build_compact_graph(
        edge_gdf
    )

    print(
        "Saving NetworkX graph..."
    )

    save_road_graph(
        graph=graph,
        output_path=GRAPH_PATH,
    )

    metrics = calculate_edge_metrics(
        edge_gdf
    )

    metrics[
        "road_features_processed"
    ] = "chunked"

    metrics[
        "chunk_size"
    ] = CHUNK_SIZE

    metrics[
        "coordinate_precision_m"
    ] = 1.0

    save_metrics(
        metrics=metrics,
        output_path=METRICS_PATH,
    )

    print("=" * 80)
    print("ROAD GRAPH COMPLETE")
    print("=" * 80)

    print(
        f"Road segments: {metrics['road_segments']:,}"
    )

    print(
        f"Nodes: {metrics['nodes']:,}"
    )

    print(
        f"Edges: {metrics['undirected_edges']:,}"
    )

    print(
        f"Connected components: {metrics['connected_components']:,}"
    )

    print(
        f"Largest component nodes: {metrics['largest_component_nodes']:,}"
    )

    print(
        f"Total road length: {metrics['total_road_length_km']:.2f} km"
    )

    print(
        f"Median segment: {metrics['median_segment_length_m']:.2f} m"
    )

    print(
        f"Mean speed: {metrics['mean_speed_kmh']:.2f} km/h"
    )

    print(
        f"Edges: {EDGE_PATH}"
    )

    print(
        f"Graph: {GRAPH_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
