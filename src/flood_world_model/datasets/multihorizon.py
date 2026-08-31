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
]

PRECIP = 0
PRECIP_3D = 1
PRECIP_7D = 2
PRECIP_LOG1P = 3
PRECIP_MISSING = 4
DISCHARGE = 5


def load_normalization(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_array(
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
        return values.astype(np.float32)

    mean = float(stats["mean"])
    std = max(
        float(stats["std"]),
        1e-8,
    )

    return (
        (values - mean) / std
    ).astype(np.float32)


def build_training_normalization(
    dynamic_ds: xr.Dataset,
    static_ds: xr.Dataset,
    train_end_index: int,
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
        values = dynamic_ds[name].isel(
            time=slice(
                0,
                train_end_index,
            )
        ).values.astype(np.float32)

        values = np.nan_to_num(
            values,
            nan=np.nan,
            posinf=np.nan,
            neginf=np.nan,
        )

        if dynamic_types[name] == "binary":
            normalization[name] = {
                "type": "binary",
            }
            continue

        finite = values[
            np.isfinite(values)
        ]

        if finite.size == 0:
            raise RuntimeError(
                f"No finite training values for {name}."
            )

        normalization[name] = {
            "type": "continuous",
            "mean": float(
                np.mean(finite)
            ),
            "std": float(
                max(
                    np.std(finite),
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
    }

    for name in STATIC_VARIABLES:
        values = static_ds[name].values.astype(
            np.float32
        )

        if static_types[name] in {
            "binary",
            "categorical",
        }:
            normalization[
                f"static_{name}"
            ] = {
                "type": static_types[name]
            }
            continue

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
                f"No finite training values for static_{name}."
            )

        normalization[
            f"static_{name}"
        ] = {
            "type": "continuous",
            "mean": float(
                np.mean(finite)
            ),
            "std": float(
                max(
                    np.std(finite),
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


class MultiHorizonFloodDataset(Dataset):
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
        self.history_length = history_length
        self.horizon = horizon

        self.normalization = load_normalization(
            normalization_path
        )

        dynamic_ds = xr.open_zarr(
            dynamic_path,
            consolidated=True,
        )

        self.times = dynamic_ds.time.values

        self.dynamic = np.stack(
            [
                normalize_array(
                    name,
                    dynamic_ds[name].values,
                    self.normalization,
                )
                for name in DYNAMIC_VARIABLES
            ],
            axis=1,
        ).astype(np.float32)

        self.raw_precip = (
            dynamic_ds[
                "precipitation"
            ].values.astype(np.float32)
        )

        self.valid = (
            dynamic_ds[
                "glofas_discharge_valid_t"
            ].values.astype(np.float32)
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

            stats = self.normalization[
                f"static_{name}"
            ]

            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if stats.get("type") not in {
                "binary",
                "categorical",
            }:
                mean = float(
                    stats["mean"]
                )
                std = max(
                    float(stats["std"]),
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

        static_ds.close()

        n = self.dynamic.shape[0]

        first = (
            start_index
            + history_length
        )

        last = (
            end_index
            - horizon
            + 1
        )

        if first >= last:
            raise ValueError(
                "Invalid dataset range. "
                f"start_index={start_index}, "
                f"end_index={end_index}"
            )

        self.indices = np.arange(
            first,
            last,
            dtype=np.int64,
        )

        if self.indices[-1] >= n:
            raise ValueError(
                "Dataset index exceeds "
                "dynamic dataset."
            )

        if not np.isfinite(
            self.dynamic
        ).all():
            raise RuntimeError(
                "Dynamic array contains NaN/Inf."
            )

        if not np.isfinite(
            self.static
        ).all():
            raise RuntimeError(
                "Static array contains NaN/Inf."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _future_forcing(
        self,
        target_start: int,
    ) -> np.ndarray:
        forcing = self.dynamic[
            target_start:
            target_start + self.horizon,
            :5,
        ]

        return forcing.astype(
            np.float32
        )

    def __getitem__(
        self,
        item: int,
    ) -> dict[str, torch.Tensor]:

        target_start = int(
            self.indices[item]
        )

        history_start = (
            target_start
            - self.history_length
        )

        history = self.dynamic[
            history_start:
            target_start
        ]

        target = self.dynamic[
            target_start:
            target_start + self.horizon,
            DISCHARGE,
        ]

        future_forcing = (
            self._future_forcing(
                target_start
            )
        )

        initial_discharge = (
            self.dynamic[
                target_start - 1,
                DISCHARGE,
            ]
        )

        glofas_valid = self.valid[
            target_start:
            target_start + self.horizon
        ]

        target_mask = (
            (
                glofas_valid
                > 0.5
            )
            & (
                self.river_mask[None, ...]
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
                target_mask
            ),
            "target_start": torch.tensor(
                target_start,
                dtype=torch.long,
            ),
        }