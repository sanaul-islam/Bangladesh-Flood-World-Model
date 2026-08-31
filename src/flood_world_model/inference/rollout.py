from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from flood_world_model.inference.predictor import V0Predictor


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


class V0Rollout:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

        self.dynamic_path = self.project_root / "data/features/dynamic_core_v2.zarr"
        self.static_path = self.project_root / "data/features/static_v3.zarr"
        self.normalization_path = self.project_root / "data/features/training_v3/normalization.json"

        with open(self.normalization_path, "r", encoding="utf-8") as f:
            self.normalization = json.load(f)

        self.predictor = V0Predictor(self.project_root)

    def load_static(self):
        static_ds = xr.open_zarr(self.static_path, consolidated=True)

        arrays = []

        for variable in STATIC_VARIABLES:
            values = static_ds[variable].values.astype(np.float32)
            values = self.predictor.normalize_static(variable, values)
            arrays.append(values)

        static_array = np.stack(arrays, axis=0).astype(np.float32)

        lat = static_ds.lat.values
        lon = static_ds.lon.values

        static_ds.close()

        return static_array, lat, lon

    def build_history(self, end_index: int):
        dynamic_ds = xr.open_zarr(self.dynamic_path, consolidated=True)

        start_index = end_index - 14

        if start_index < 0:
            raise ValueError("Not enough history for a 14-day window.")

        arrays = []

        for variable in DYNAMIC_VARIABLES:
            values = dynamic_ds[variable].isel(time=slice(start_index, end_index)).values.astype(np.float32)
            values = self.predictor.normalize_dynamic(variable, values)
            arrays.append(values)

        history = np.stack(arrays, axis=1).astype(np.float32)

        history_dates = dynamic_ds.time.values[start_index:end_index]
        lat = dynamic_ds.lat.values
        lon = dynamic_ds.lon.values

        dynamic_ds.close()

        return history, history_dates, lat, lon

    def rollout(self, end_index: int, days: int = 7):
        history, history_dates, lat, lon = self.build_history(end_index)
        static_array, _, _ = self.load_static()

        current = history.copy()

        predictions = []

        for day in range(days):
            predicted_normalized = self.predictor.predict(current, static_array)
            predicted_physical = self.predictor.denormalize_discharge(predicted_normalized)

            predictions.append(predicted_physical)

            next_step = current[-1].copy()

            next_step[5] = predicted_normalized

            current = np.concatenate(
                [
                    current[1:],
                    next_step[None],
                ],
                axis=0,
            )

            print(f"Rollout day {day + 1}/{days} complete.")

        predictions = np.stack(predictions, axis=0).astype(np.float32)

        return {
            "predictions": predictions,
            "history_dates": history_dates,
            "lat": lat,
            "lon": lon,
        }

    def save(self, result, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        forecast_days = np.arange(1, result["predictions"].shape[0] + 1)

        dataset = xr.Dataset(
            {
                "predicted_river_discharge": (
                    ("forecast_day", "lat", "lon"),
                    result["predictions"],
                )
            },
            coords={
                "forecast_day": forecast_days,
                "lat": result["lat"],
                "lon": result["lon"],
            },
        )

        dataset["predicted_river_discharge"].attrs["units"] = "m3 s-1"
        dataset["predicted_river_discharge"].attrs["model"] = "World Model V0"
        dataset["predicted_river_discharge"].attrs["description"] = "Seven-day autoregressive discharge forecast."

        dataset.to_netcdf(output_path)

        dataset.close()

        print(f"Saved forecast: {output_path}")