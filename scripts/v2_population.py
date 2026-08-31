from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


DYNAMIC_VARIABLES = [
    "precipitation",
    "precip_3d",
    "precip_7d",
    "precip_log1p",
    "precip_missing",
    "river_discharge",
]

STATIC_VARIABLES = [
    "elevation",
    "slope_degrees",
    "flow_accumulation",
    "river_mask",
    "river_distance_km",
    "landcover",
    "soil_clay",
    "soil_silt",
    "soil_sand",
    "soil_organic_carbon",
    "land_mask",
    "population_density",
]

DISCHARGE_CHANNEL = 5


def load_normalization(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_dynamic(
    name: str,
    values: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    stats = normalization[name]

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if stats.get("type") in {
        "binary",
        "categorical",
    }:
        return values.astype(
            np.float32
        )

    mean = float(
        stats["mean"]
    )

    std = max(
        float(stats["std"]),
        1e-8,
    )

    return (
        (values - mean) / std
    ).astype(np.float32)


def build_normalization(
    dynamic_ds: xr.Dataset,
    static_ds: xr.Dataset,
    train_end: int,
    output_path: str | Path,
) -> dict[str, Any]:
    normalization: dict[str, Any] = {}

    dynamic_types = {
        "precipitation": "continuous",
        "precip_3d": "continuous",
        "precip_7d": "continuous",
        "precip_log1p": "continuous",
        "precip_missing": "binary",
        "river_discharge": "continuous",
    }

    for name in DYNAMIC_VARIABLES:
        if dynamic_types[name] == "binary":
            normalization[name] = {
                "type": "binary",
            }
            continue

        values = (
            dynamic_ds[name]
            .isel(
                time=slice(
                    0,
                    train_end,
                )
            )
            .values
            .astype(np.float32)
        )

        values = np.nan_to_num(
            values,
            nan=np.nan,
            posinf=np.nan,
            neginf=np.nan,
        )

        finite = values[
            np.isfinite(values)
        ]

        if finite.size == 0:
            raise RuntimeError(
                f"No finite training values for {name}"
            )

        normalization[name] = {
            "type": "continuous",
            "mean": float(
                finite.mean()
            ),
            "std": float(
                max(
                    finite.std(),
                    1e-8,
                )
            ),
        }

    static_types = {
        "elevation": "continuous",
        "slope_degrees": "continuous",
        "flow_accumulation": "continuous",
        "river_mask": "binary",
        "river_distance_km": "continuous",
        "landcover": "categorical",
        "soil_clay": "continuous",
        "soil_silt": "continuous",
        "soil_sand": "continuous",
        "soil_organic_carbon": "continuous",
        "land_mask": "binary",
        "population_density": "continuous",
    }

    for name in STATIC_VARIABLES:
        values = (
            static_ds[name]
            .values
            .astype(np.float32)
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if static_types[name] in {
            "binary",
            "categorical",
        }:
            normalization[
                f"static_{name}"
            ] = {
                "type": static_types[name],
            }
            continue

        if name == "population_density":
            values_for_stats = np.log1p(
                np.maximum(
                    values,
                    0.0,
                )
            )
        else:
            values_for_stats = values

        finite = values_for_stats[
            np.isfinite(values_for_stats)
        ]

        if finite.size == 0:
            raise RuntimeError(
                f"No finite values for static_{name}"
            )

        normalization[
            f"static_{name}"
        ] = {
            "type": "continuous",
            "transform": (
                "log1p"
                if name == "population_density"
                else "identity"
            ),
            "mean": float(
                finite.mean()
            ),
            "std": float(
                max(
                    finite.std(),
                    1e-8,
                )
            ),
        }

    output_path = Path(
        output_path
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
            normalization,
            file,
            indent=2,
        )

    return normalization


class V2PopulationDataset(Dataset):
    def __init__(
        self,
        dynamic_path: str | Path,
        static_path: str | Path,
        normalization_path: str | Path,
        start_index: int,
        end_index: int,
        history_length: int = 14,
        horizon: int = 7,
    ) -> None:
        self.history_length = (
            history_length
        )

        self.horizon = horizon

        self.normalization = (
            load_normalization(
                normalization_path
            )
        )

        dynamic_ds = xr.open_zarr(
            dynamic_path,
            consolidated=True,
        )

        self.time = (
            dynamic_ds.time.values
        )

        dynamic_arrays = []

        for name in DYNAMIC_VARIABLES:
            values = (
                dynamic_ds[name]
                .values
                .astype(np.float32)
            )

            dynamic_arrays.append(
                normalize_dynamic(
                    name,
                    values,
                    self.normalization,
                )
            )

        self.dynamic = np.stack(
            dynamic_arrays,
            axis=1,
        ).astype(np.float32)

        self.valid = (
            dynamic_ds[
                "glofas_discharge_valid_t"
            ]
            .values
            .astype(np.float32)
        )

        dynamic_ds.close()

        static_ds = xr.open_zarr(
            static_path,
            consolidated=True,
        )

        static_arrays = []

        for name in STATIC_VARIABLES:
            values = (
                static_ds[name]
                .values
                .astype(np.float32)
            )

            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            stats = self.normalization[
                f"static_{name}"
            ]

            if name == "population_density":
                values = np.log1p(
                    np.maximum(
                        values,
                        0.0,
                    )
                )

            if stats.get("type") not in {
                "binary",
                "categorical",
            }:
                mean = float(
                    stats["mean"]
                )

                std = max(
                    float(
                        stats["std"]
                    ),
                    1e-8,
                )

                values = (
                    values - mean
                ) / std

            static_arrays.append(
                values.astype(
                    np.float32
                )
            )

        self.static = np.stack(
            static_arrays,
            axis=0,
        ).astype(np.float32)

        self.river_mask = (
            static_ds[
                "river_mask"
            ]
            .values
            .astype(np.float32)
        )

        self.river_mask = (
            self.river_mask > 0.5
        ).astype(np.float32)

        self.population_density = (
            static_ds[
                "population_density"
            ]
            .values
            .astype(np.float32)
        )

        static_ds.close()

        expected_static = (
            12,
            self.static.shape[1],
            self.static.shape[2],
        )

        if self.static.shape != expected_static:
            raise RuntimeError(
                f"Expected static shape {expected_static}, got {self.static.shape}"
            )

        if not np.isfinite(
            self.dynamic
        ).all():
            raise RuntimeError(
                "Dynamic data contains NaN or Inf."
            )

        if not np.isfinite(
            self.static
        ).all():
            raise RuntimeError(
                "Static data contains NaN or Inf."
            )

        first = (
            start_index
            + history_length
        )

        last = (
            end_index
        )

        self.indices = np.arange(
            first,
            last,
            dtype=np.int64,
        )

        if len(self.indices) == 0:
            raise RuntimeError(
                "No samples available for this split."
            )

    def __len__(self) -> int:
        return len(
            self.indices
        )

    def __getitem__(
        self,
        item: int,
    ) -> dict[str, torch.Tensor]:
        target_index = int(
            self.indices[item]
        )

        history_start = (
            target_index
            - self.history_length
        )

        history = self.dynamic[
            history_start:
            target_index
        ]

        future_forcing = self.dynamic[
            target_index:
            target_index + self.horizon,
            :5,
        ]

        target = self.dynamic[
            target_index:
            target_index + self.horizon,
            DISCHARGE_CHANNEL,
        ]

        initial_discharge = (
            self.dynamic[
                target_index - 1,
                DISCHARGE_CHANNEL,
            ]
        )

        valid = (
            self.valid[
                target_index:
                target_index + self.horizon
            ]
            > 0.5
        )

        mask = (
            valid
            & (
                self.river_mask[None]
                > 0.5
            )
        ).astype(np.float32)

        return {
            "history": torch.from_numpy(
                history
            ),
            "static": torch.from_numpy(
                self.static
            ),
            "future_forcing": torch.from_numpy(
                future_forcing
            ),
            "initial_discharge": torch.from_numpy(
                initial_discharge
            ),
            "target": torch.from_numpy(
                target
            ),
            "mask": torch.from_numpy(
                mask
            ),
            "index": torch.tensor(
                target_index,
                dtype=torch.long,
            ),
        }
    