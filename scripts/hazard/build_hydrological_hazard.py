
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT / "src"
    ),
)

from flood_world_model.datasets.multihorizon import (
    MultiHorizonFloodDataset,
)

from flood_world_model.hazard.hydrological import (
    build_hydrological_hazard,
    save_hydrological_hazard,
)


DYNAMIC_PATH = (
    PROJECT_ROOT
    / "data/features/dynamic_core_v2.zarr"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "data/features/static_v3.zarr"
)

NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/features/training_v3/v2_normalization.json"
)

UNCERTAINTY_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/v2_population_uncertainty.nc"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_hydrological_hazard.nc"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/"
    "v2_population_hydrological_hazard.json"
)

HISTORY_LENGTH = 14
HORIZON = 7


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("HYDROLOGICAL HAZARD LAYER")
    print("=" * 80)

    required = [
        DYNAMIC_PATH,
        STATIC_PATH,
        NORMALIZATION_PATH,
        UNCERTAINTY_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    print(
        "Building canonical test dataset..."
    )

    dataset = MultiHorizonFloodDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=3544,
        end_index=4170,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    sample_indices = (
        dataset.indices.copy()
    )

    print(
        f"Forecast samples: {len(sample_indices)}"
    )

    if len(sample_indices) != 606:
        raise RuntimeError(
            f"Expected 606 canonical test samples, got {len(sample_indices)}"
        )

    print(
        "Building hydrological hazard..."
    )

    hazard_ds = build_hydrological_hazard(
        uncertainty_path=UNCERTAINTY_PATH,
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        sample_indices=sample_indices,
        weights={
            "discharge": 0.40,
            "uncertainty": 0.20,
            "rainfall": 0.15,
            "elevation": 0.10,
            "river_distance": 0.10,
            "slope": 0.05,
        },
    )

    summary = save_hydrological_hazard(
        dataset=hazard_ds,
        output_path=OUTPUT_PATH,
        metrics_path=METRICS_PATH,
    )

    hazard_values = (
        hazard_ds[
            "hydrological_hazard_score"
        ].values
    )

    finite = hazard_values[
        np.isfinite(
            hazard_values
        )
    ]

    print("=" * 80)
    print("HYDROLOGICAL HAZARD COMPLETE")
    print("=" * 80)

    print(
        f"Finite hazard values: {summary['finite_values']}"
    )

    print(
        f"Hazard min: {summary['min']}"
    )

    print(
        f"Hazard mean: {summary['mean']}"
    )

    print(
        f"Hazard median: {summary['median']}"
    )

    print(
        f"Hazard max: {summary['max']}"
    )

    print(
        f"Forecast: {OUTPUT_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )

    hazard_ds.close()


if __name__ == "__main__":
    main()
