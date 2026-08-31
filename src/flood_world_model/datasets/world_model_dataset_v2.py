
from __future__ import annotations

import json

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from flood_world_model.utils.paths import FEATURES_DIR, TRAINING_DIR


class FloodWorldModelDataset(Dataset):
    def __init__(self, split: str = "train", history_days: int = 14, forecast_days: int = 1):
        self.history_days = history_days
        self.forecast_days = forecast_days

        training = TRAINING_DIR
        dynamic_path = FEATURES_DIR / "dynamic_core_v2.zarr"
        static_path = FEATURES_DIR / "static_v3.zarr"

        index_paths = {
            "train": training / "train_indices.npy",
            "val": training / "val_indices.npy",
            "test": training / "test_indices.npy",
            "live": training / "live_indices.npy",
        }

        if split not in index_paths:
            raise ValueError(f"Unknown split: {split}")

        self.indices = np.load(index_paths[split]).astype(np.int64)

        with open(training / "normalization.json", "r", encoding="utf-8") as f:
            self.norm = json.load(f)

        self.dynamic_vars = [
            "precipitation",
            "precip_3d",
            "precip_7d",
            "precip_log1p",
            "precip_missing",
            "river_discharge",
        ]

        self.static_vars = [
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

        self.target_vars = [
            "river_discharge",
        ]

        print("Loading dynamic data into RAM...")
        dynamic = xr.open_zarr(dynamic_path, consolidated=True)
        self.dynamic = {}

        for name in self.dynamic_vars:
            print(f"  Loading {name}...")
            self.dynamic[name] = dynamic[name].values.astype(np.float32)

        dynamic.close()

        print("Loading static data into RAM...")
        static = xr.open_zarr(static_path, consolidated=True)
        self.static = {}

        for name in self.static_vars:
            print(f"  Loading {name}...")
            self.static[name] = static[name].values.astype(np.float32)

        static.close()

        print("Normalizing dynamic arrays...")

        for name in self.dynamic_vars:
            stats = self.norm[name]

            if stats.get("type") == "binary":
                self.dynamic[name] = np.nan_to_num(self.dynamic[name], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                mean = float(stats["mean"])
                std = max(float(stats["std"]), 1e-8)
                self.dynamic[name] = (self.dynamic[name] - mean) / std
                self.dynamic[name] = np.nan_to_num(self.dynamic[name], nan=0.0, posinf=0.0, neginf=0.0)

        print("Normalizing static arrays...")

        for name in self.static_vars:
            stats = self.norm[f"static_{name}"]
            kind = stats.get("type")

            if kind == "binary" or kind == "categorical":
                self.static[name] = np.nan_to_num(self.static[name], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                mean = float(stats["mean"])
                std = max(float(stats["std"]), 1e-8)
                self.static[name] = (self.static[name] - mean) / std
                self.static[name] = np.nan_to_num(self.static[name], nan=0.0, posinf=0.0, neginf=0.0)

        self.static_tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.stack([self.static[v] for v in self.static_vars], axis=0),
                dtype=np.float32,
            )
        )

        print("Dataset loaded into RAM.")
        print(f"Samples: {len(self.indices):,}")
        print(f"Dynamic variables: {len(self.dynamic_vars)}")
        print(f"Static variables: {len(self.static_vars)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        start = int(self.indices[index])
        end = start + self.history_days

        target_start = end
        target_end = end + self.forecast_days

        x = np.stack(
            [self.dynamic[v][start:end] for v in self.dynamic_vars],
            axis=1,
        )

        y = np.stack(
            [self.dynamic[v][target_start:target_end] for v in self.target_vars],
            axis=1,
        )

        target_mask = np.isfinite(
            self.dynamic["river_discharge"][target_start:target_end]
        ).astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            self.static_tensor,
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(np.ascontiguousarray(target_mask[:, None])),
        )


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING RAM-CACHED WORLD MODEL DATASET")
    print("=" * 80)

    dataset = FloodWorldModelDataset("train")

    x, static, y, mask = dataset[0]

    print("X:", x.shape, x.dtype)
    print("Static:", static.shape, static.dtype)
    print("Y:", y.shape, y.dtype)
    print("Mask:", mask.shape, mask.dtype)
    print("X finite:", bool(torch.isfinite(x).all()))
    print("Static finite:", bool(torch.isfinite(static).all()))
    print("Y finite:", bool(torch.isfinite(y).all()))
    print("Mask finite:", bool(torch.isfinite(mask).all()))
    print("✅ Dataset test complete.")
