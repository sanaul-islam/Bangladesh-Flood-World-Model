
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT / "src"
    ),
)

from flood_world_model.risk.population import (
    build_population_risk,
    save_population_risk,
)


HAZARD_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_hydrological_hazard.nc"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "data/features/static_v3.zarr"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_population_risk.nc"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/"
    "v2_population_population_risk.json"
)


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("POPULATION EXPOSURE / RISK LAYER")
    print("=" * 80)

    required = [
        HAZARD_PATH,
        STATIC_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    print(
        f"Hazard source: {HAZARD_PATH}"
    )

    print(
        f"Population source: {STATIC_PATH}"
    )

    print(
        "Building population exposure..."
    )

    dataset = build_population_risk(
        hazard_path=HAZARD_PATH,
        static_path=STATIC_PATH,
        population_weight=1.0,
    )

    summary = save_population_risk(
        dataset=dataset,
        output_path=OUTPUT_PATH,
        metrics_path=METRICS_PATH,
    )

    exposure = (
        dataset[
            "population_exposure_index"
        ]
        .values
    )

    finite = exposure[
        np.isfinite(exposure)
    ]

    print("=" * 80)
    print("POPULATION RISK COMPLETE")
    print("=" * 80)

    print(
        f"Finite exposure values: {summary['finite_values']}"
    )

    print(
        f"Mean exposure index: {summary['mean_exposure_index']}"
    )

    print(
        f"Median exposure index: {summary['median_exposure_index']}"
    )

    print(
        f"Maximum exposure index: {summary['max_exposure_index']}"
    )

    print(
        f"High-exposure fraction: {summary['high_exposure_fraction']}"
    )

    print(
        f"Forecast: {OUTPUT_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )

    dataset.close()


if __name__ == "__main__":
    main()
