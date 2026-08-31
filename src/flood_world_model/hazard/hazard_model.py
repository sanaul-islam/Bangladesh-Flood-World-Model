from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


class HazardEngine:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

        self.static_path = (
            self.project_root
            / "data/features/static_v3.zarr"
        )

    @staticmethod
    def robust_scale(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        valid = np.isfinite(values)

        if not valid.any():
            return np.zeros_like(
                values,
                dtype=np.float32,
            )

        low = np.percentile(
            values[valid],
            10,
        )

        high = np.percentile(
            values[valid],
            90,
        )

        if high <= low:
            return np.zeros_like(
                values,
                dtype=np.float32,
            )

        scaled = (
            values - low
        ) / (
            high - low
        )

        return np.clip(
            scaled,
            0.0,
            1.0,
        ).astype(np.float32)

    def build(
        self,
        forecast_path: Path,
        output_path: Path,
    ) -> None:
        print("=" * 80)
        print("BUILDING HYDROLOGICAL HAZARD")
        print("=" * 80)

        if not forecast_path.exists():
            raise FileNotFoundError(
                f"Forecast not found: {forecast_path}"
            )

        if not self.static_path.exists():
            raise FileNotFoundError(
                f"Static dataset not found: {self.static_path}"
            )

        print("Loading forecast...")

        forecast = xr.open_dataset(
            forecast_path
        )

        required_forecast = [
            "predicted_river_discharge",
        ]

        for variable in required_forecast:
            if variable not in forecast:
                forecast.close()
                raise KeyError(
                    f"Missing forecast variable: {variable}"
                )

        discharge = forecast[
            "predicted_river_discharge"
        ].values.astype(np.float32)

        forecast_lat = forecast.lat.values
        forecast_lon = forecast.lon.values
        forecast_days = forecast.forecast_day.values

        if discharge.ndim != 3:
            forecast.close()
            raise RuntimeError(
                f"Expected forecast shape [day,lat,lon], got {discharge.shape}"
            )

        print(
            f"Forecast shape: {discharge.shape}"
        )

        print("Loading static features...")

        static = xr.open_zarr(
            self.static_path,
            consolidated=True,
        )

        required_static = [
            "elevation",
            "river_distance_km",
            "river_mask",
            "land_mask",
        ]

        for variable in required_static:
            if variable not in static:
                forecast.close()
                static.close()
                raise KeyError(
                    f"Missing static variable: {variable}"
                )

        elevation = static[
            "elevation"
        ].values.astype(np.float32)

        river_distance = static[
            "river_distance_km"
        ].values.astype(np.float32)

        river_mask = static[
            "river_mask"
        ].values.astype(np.float32)

        land_mask = static[
            "land_mask"
        ].values.astype(np.float32)

        static_lat = static.lat.values
        static_lon = static.lon.values

        static.close()

        if not np.allclose(
            forecast_lat,
            static_lat,
        ):
            forecast.close()
            raise RuntimeError(
                "Forecast latitude grid does not match static grid."
            )

        if not np.allclose(
            forecast_lon,
            static_lon,
        ):
            forecast.close()
            raise RuntimeError(
                "Forecast longitude grid does not match static grid."
            )

        elevation = np.nan_to_num(
            elevation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        river_distance = np.nan_to_num(
            river_distance,
            nan=9999.0,
            posinf=9999.0,
            neginf=9999.0,
        )

        river_mask = np.nan_to_num(
            river_mask,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        land_mask = np.nan_to_num(
            land_mask,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        print("Building static risk components...")

        distance_risk = np.exp(
            -river_distance / 20.0
        ).astype(np.float32)

        low_elevation_risk = np.exp(
            -np.maximum(
                elevation,
                0.0,
            ) / 20.0
        ).astype(np.float32)

        distance_risk *= land_mask
        low_elevation_risk *= land_mask

        hazard_maps = []

        print("Computing hazard for each forecast day...")

        for index in range(
            discharge.shape[0]
        ):
            daily_discharge = discharge[
                index
            ]

            discharge_risk = self.robust_scale(
                daily_discharge
            )

            hazard = (
                0.65 * discharge_risk
                + 0.20 * distance_risk
                + 0.15 * low_elevation_risk
            )

            hazard *= land_mask

            river_influence = np.maximum(
                river_mask,
                0.25,
            )

            hazard *= river_influence

            hazard = np.clip(
                hazard,
                0.0,
                1.0,
            ).astype(np.float32)

            if not np.isfinite(
                hazard
            ).all():
                forecast.close()
                raise RuntimeError(
                    f"Hazard contains NaN/Inf on forecast day {index + 1}"
                )

            hazard_maps.append(
                hazard
            )

            print(
                f"Day {index + 1}: "
                f"min={hazard.min():.4f} "
                f"max={hazard.max():.4f} "
                f"mean={hazard.mean():.4f}"
            )

        hazard_maps = np.stack(
            hazard_maps,
            axis=0,
        ).astype(np.float32)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = xr.Dataset(
            {
                "hazard_score": (
                    (
                        "forecast_day",
                        "lat",
                        "lon",
                    ),
                    hazard_maps,
                )
            },
            coords={
                "forecast_day": forecast_days,
                "lat": forecast_lat,
                "lon": forecast_lon,
            },
            attrs={
                "model": "World Model V0",
                "hazard_type": "hydrological_hazard_score",
                "calibration": "uncalibrated",
                "description": (
                    "Composite hydrological hazard score "
                    "based on predicted discharge, river proximity, "
                    "elevation, and land validity."
                ),
                "discharge_weight": 0.65,
                "river_distance_weight": 0.20,
                "elevation_weight": 0.15,
                "river_distance_scale_km": 20.0,
                "elevation_scale_m": 20.0,
            },
        )

        output[
            "hazard_score"
        ].attrs["units"] = "0-1"

        output[
            "hazard_score"
        ].attrs["note"] = (
            "This is not calibrated flood probability."
        )

        output.to_netcdf(
            output_path
        )

        output.close()
        forecast.close()

        print(
            f"✅ Hazard saved: {output_path}"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    forecast_file = (
        project_root
        / "outputs/predictions/v0_7day_forecast.nc"
    )

    output_file = (
        project_root
        / "outputs/hazard/v0_7day_hazard.nc"
    )

    HazardEngine(
        project_root
    ).build(
        forecast_file,
        output_file,
    )