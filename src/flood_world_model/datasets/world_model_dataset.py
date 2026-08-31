from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class FloodWorldModelDataset(Dataset):
    """
    Lazy Zarr-backed dataset for the Bangladesh Flood World Model.

    Current V0 formulation:

        14 days of history
            ↓
        predict next 1 day

    Dynamic input:
        [T, C_dynamic, H, W]

    Static input:
        [C_static, H, W]

    Target:
        [T_future, C_target, H, W]
    """

    def __init__(
        self,
        dynamic_path: str | Path,
        static_path: str | Path,
        indices_path: str | Path,
        normalization_path: str | Path,
        dynamic_variables: list[str],
        static_variables: list[str],
        target_variables: list[str],
        history_days: int = 14,
        forecast_days: int = 1,
    ) -> None:

        self.dynamic_path = str(dynamic_path)
        self.static_path = str(static_path)

        self.indices = np.load(
            indices_path
        ).astype(np.int64)

        with open(
            normalization_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.normalization = json.load(f)

        self.dynamic_variables = dynamic_variables
        self.static_variables = static_variables
        self.target_variables = target_variables

        self.history_days = int(history_days)
        self.forecast_days = int(forecast_days)

        # Open lazily on first __getitem__.
        #
        # This is important for low-RAM machines and also
        # works better when DataLoader workers are used.
        self.dynamic: Optional[xr.Dataset] = None
        self.static: Optional[xr.Dataset] = None

        # Cache static features after first load.
        #
        # Static data is only 60 × 45, so this is tiny.
        self._static_tensor: Optional[torch.Tensor] = None

    # ========================================================
    # Lazy dataset opening
    # ========================================================

    def _open(self) -> None:

        if self.dynamic is None:
            self.dynamic = xr.open_zarr(
                self.dynamic_path,
                consolidated=True,
            )

            self._validate_dynamic()

        if self.static is None:
            self.static = xr.open_zarr(
                self.static_path,
                consolidated=True,
            )

            self._validate_static()

    # ========================================================
    # Validation
    # ========================================================

    def _validate_dynamic(self) -> None:

        assert self.dynamic is not None

        available = set(
            self.dynamic.data_vars
        )

        required = set(
            self.dynamic_variables
            + self.target_variables
        )

        missing = sorted(
            required - available
        )

        if missing:
            raise ValueError(
                "Dynamic dataset is missing variables:\n"
                + "\n".join(
                    f"  - {v}"
                    for v in missing
                )
                + "\n\nAvailable variables:\n"
                + "\n".join(
                    f"  - {v}"
                    for v in sorted(available)
                )
            )

    def _validate_static(self) -> None:

        assert self.static is not None

        available = set(
            self.static.data_vars
        )

        missing = sorted(
            set(self.static_variables)
            - available
        )

        if missing:
            raise ValueError(
                "Static dataset is missing variables:\n"
                + "\n".join(
                    f"  - {v}"
                    for v in missing
                )
            )

        if self.static.sizes["lat"] != self.dynamic.sizes["lat"]:
            raise ValueError(
                "Static/dynamic latitude dimensions differ."
            )

        if self.static.sizes["lon"] != self.dynamic.sizes["lon"]:
            raise ValueError(
                "Static/dynamic longitude dimensions differ."
            )

        if not np.allclose(
            self.static.lat.values,
            self.dynamic.lat.values,
        ):
            raise ValueError(
                "Static/dynamic latitude coordinates differ."
            )

        if not np.allclose(
            self.static.lon.values,
            self.dynamic.lon.values,
        ):
            raise ValueError(
                "Static/dynamic longitude coordinates differ."
            )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self) -> int:
        return len(self.indices)

    # ========================================================
    # Normalization
    # ========================================================

    def _normalize_dynamic(
        self,
        variable: str,
        values: np.ndarray,
    ) -> np.ndarray:

        stats = self.normalization.get(
            variable
        )

        if stats is None:
            raise KeyError(
                f"No normalization statistics for "
                f"dynamic variable '{variable}'."
            )

        feature_type = stats.get(
            "type",
            "standard",
        )

        # Binary feature.
        if feature_type == "binary":
            return values.astype(
                np.float32,
                copy=False,
            )

        mean = float(
            stats.get("mean", 0.0)
        )

        std = float(
            stats.get("std", 1.0)
        )

        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        return (
            values - mean
        ) / std

    def _normalize_static(
        self,
        variable: str,
        values: np.ndarray,
    ) -> np.ndarray:

        key = f"static_{variable}"

        stats = self.normalization.get(
            key
        )

        if stats is None:
            raise KeyError(
                f"No normalization statistics for "
                f"static variable '{variable}'."
            )

        feature_type = stats.get(
            "type",
            "standard",
        )

        # Landcover is categorical.
        #
        # Do NOT perform:
        #
        # (class - mean) / std
        #
        # because class IDs are labels, not continuous values.
        if feature_type == "categorical":
            return values.astype(
                np.float32,
                copy=False,
            )

        mean = stats.get(
            "mean"
        )

        std = stats.get(
            "std"
        )

        if mean is None:
            mean = 0.0

        if (
            std is None
            or not np.isfinite(std)
            or std < 1e-8
        ):
            std = 1.0

        return (
            values - float(mean)
        ) / float(std)

    # ========================================================
    # Static cache
    # ========================================================

    def _get_static_tensor(
        self,
    ) -> torch.Tensor:

        if self._static_tensor is not None:
            return self._static_tensor

        assert self.static is not None

        arrays = []

        for variable in self.static_variables:

            values = (
                self.static[variable]
                .values
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            values = self._normalize_static(
                variable,
                values,
            )

            arrays.append(values)

        static_array = np.stack(
            arrays,
            axis=0,
        )

        static_array = np.ascontiguousarray(
            static_array,
            dtype=np.float32,
        )

        self._static_tensor = (
            torch.from_numpy(
                static_array
            )
        )

        return self._static_tensor

    # ========================================================
    # Get one sample
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ):

        self._open()

        assert self.dynamic is not None

        start = int(
            self.indices[index]
        )

        input_start = start
        input_end = (
            start
            + self.history_days
        )

        target_start = input_end
        target_end = (
            target_start
            + self.forecast_days
        )

        # ====================================================
        # DYNAMIC INPUT
        # ====================================================

        dynamic_arrays = []

        for variable in self.dynamic_variables:

            array = (
                self.dynamic[variable]
                .isel(
                    time=slice(
                        input_start,
                        input_end,
                    )
                )
                .values
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            array = self._normalize_dynamic(
                variable,
                array,
            )

            dynamic_arrays.append(
                array
            )

        # Current arrays are:
        #
        # [T, H, W]
        #
        # Stack:
        #
        # [C, T, H, W]

        dynamic_array = np.stack(
            dynamic_arrays,
            axis=0,
        )

        # Convert:
        #
        # [C, T, H, W]
        #
        # →
        #
        # [T, C, H, W]

        dynamic_array = np.transpose(
            dynamic_array,
            (1, 0, 2, 3),
        )

        dynamic_array = np.ascontiguousarray(
            dynamic_array,
            dtype=np.float32,
        )

        # ====================================================
        # STATIC INPUT
        # ====================================================

        static_tensor = self._get_static_tensor()

        # ====================================================
        # TARGET
        # ====================================================

        target_arrays = []

        for variable in self.target_variables:

            array = (
                self.dynamic[variable]
                .isel(
                    time=slice(
                        target_start,
                        target_end,
                    )
                )
                .values
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            array = self._normalize_dynamic(
                variable,
                array,
            )

            target_arrays.append(
                array
            )

        # [C_target, T, H, W]

        target_array = np.stack(
            target_arrays,
            axis=0,
        )

        # [T, C_target, H, W]

        target_array = np.transpose(
            target_array,
            (1, 0, 2, 3),
        )

        target_array = np.ascontiguousarray(
            target_array,
            dtype=np.float32,
        )

        # ====================================================
        # Convert to tensors
        # ====================================================

        x = torch.from_numpy(
            dynamic_array
        )

        y = torch.from_numpy(
            target_array
        )

        # Clone static because it is cached.
        #
        # This avoids accidental in-place modification by
        # later code.

        static_x = static_tensor

        return (
            x,
            static_x,
            y,
        )


# ============================================================
# Helper: current V0 dataset
# ============================================================

def create_v0_dataset(
    split: str = "train",
) -> FloodWorldModelDataset:

    """
    Convenience constructor for the current project layout.
    """

    dynamic_variables = [
        "precipitation",
        "precip_3d",
        "precip_7d",
        "precip_log1p",
        "precip_missing",
        "river_discharge",
    ]

    static_variables = [
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
    ]

    target_variables = [
        "river_discharge",
    ]

    split_files = {
        "train": (
            "data/features/training/"
            "train_indices.npy"
        ),
        "val": (
            "data/features/training/"
            "val_indices.npy"
        ),
        "test": (
            "data/features/training/"
            "test_indices.npy"
        ),
    }

    if split not in split_files:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Use train, val, or test."
        )

    return FloodWorldModelDataset(
        dynamic_path=(
            "data/features/dynamic_core.zarr"
        ),
        static_path=(
            "data/features/static.zarr"
        ),
        indices_path=split_files[split],
        normalization_path=(
            "data/features/training/"
            "normalization.json"
        ),
        dynamic_variables=dynamic_variables,
        static_variables=static_variables,
        target_variables=target_variables,
        history_days=14,
        forecast_days=1,
    )


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("TESTING FLOOD WORLD MODEL DATASET")
    print("=" * 80)

    dataset = create_v0_dataset(
        "train"
    )

    print(
        f"Number of samples: "
        f"{len(dataset):,}"
    )

    x, static, y = dataset[0]

    print()
    print(
        "Dynamic input:",
        x.shape,
        x.dtype,
    )

    print(
        "Static input:",
        static.shape,
        static.dtype,
    )

    print(
        "Target:",
        y.shape,
        y.dtype,
    )

    print()

    print(
        "Dynamic finite:",
        bool(
            torch.isfinite(x).all()
        ),
    )

    print(
        "Static finite:",
        bool(
            torch.isfinite(static).all()
        ),
    )

    print(
        "Target finite:",
        bool(
            torch.isfinite(y).all()
        ),
    )

    print()
    print("✅ Dataset test complete.")