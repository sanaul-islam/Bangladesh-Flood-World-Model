from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from flood_world_model.models.world_model import FloodWorldModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DYNAMIC_PATH = PROJECT_ROOT / "data/features/dynamic_core_v2.zarr"
STATIC_PATH = PROJECT_ROOT / "data/features/static_v3.zarr"
NORMALIZATION_PATH = PROJECT_ROOT / "data/features/training_v3/normalization.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/world_model_v0_best.pt"

FORECAST_PATH = PROJECT_ROOT / "outputs/predictions/v0_7day_forecast.nc"
METRICS_PATH = PROJECT_ROOT / "outputs/metrics/v0_7day_rollout_metrics.json"

HISTORY_LENGTH = 14
HORIZON = 7

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


def check_files() -> None:
    required = [
        DYNAMIC_PATH,
        STATIC_PATH,
        NORMALIZATION_PATH,
        CHECKPOINT_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_normalization() -> dict[str, Any]:
    with NORMALIZATION_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize(name: str, values: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    stats = normalization[name]

    values = np.asarray(values, dtype=np.float32)

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if stats.get("type") in {"binary", "categorical"}:
        return values.astype(np.float32)

    mean = float(stats["mean"])
    std = max(float(stats["std"]), 1e-8)

    return ((values - mean) / std).astype(np.float32)


def denormalize(name: str, values: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    stats = normalization[name]

    values = np.asarray(values, dtype=np.float32)

    if stats.get("type") in {"binary", "categorical"}:
        return values.astype(np.float32)

    mean = float(stats["mean"])
    std = max(float(stats["std"]), 1e-8)

    return (values * std + mean).astype(np.float32)


def load_static(
    normalization: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_zarr(STATIC_PATH, consolidated=True)

    arrays = []

    for name in STATIC_VARIABLES:
        if name not in ds:
            ds.close()
            raise KeyError(f"Missing static variable: {name}")

        stats = normalization[f"static_{name}"]

        values = ds[name].values.astype(np.float32)

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if stats.get("type") not in {"binary", "categorical"}:
            mean = float(stats["mean"])
            std = max(float(stats["std"]), 1e-8)
            values = (values - mean) / std

        arrays.append(values.astype(np.float32))

    static = np.stack(arrays, axis=0).astype(np.float32)

    river_mask = ds["river_mask"].values.astype(np.float32)

    river_mask = np.nan_to_num(
        river_mask,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    river_mask = (river_mask > 0.5).astype(np.float32)

    lat = ds["lat"].values
    lon = ds["lon"].values

    ds.close()

    if not np.isfinite(static).all():
        raise RuntimeError("Static input contains NaN or Inf.")

    return static, river_mask, lat, lon


def load_history(
    ds: xr.Dataset,
    end_index: int,
    normalization: dict[str, Any],
) -> np.ndarray:
    if end_index < HISTORY_LENGTH:
        raise ValueError(f"end_index={end_index} is too small for {HISTORY_LENGTH} days of history.")

    arrays = []

    for name in DYNAMIC_VARIABLES:
        values = ds[name].isel(
            time=slice(
                end_index - HISTORY_LENGTH,
                end_index,
            )
        ).values.astype(np.float32)

        arrays.append(normalize(name, values, normalization))

    history = np.stack(arrays, axis=1).astype(np.float32)

    expected_shape = (
        HISTORY_LENGTH,
        len(DYNAMIC_VARIABLES),
        ds.sizes["lat"],
        ds.sizes["lon"],
    )

    if history.shape != expected_shape:
        raise RuntimeError(f"Unexpected history shape: {history.shape}; expected {expected_shape}")

    if not np.isfinite(history).all():
        raise RuntimeError("Dynamic history contains NaN or Inf.")

    return history


def build_model(normalization: dict[str, Any]) -> tuple[torch.nn.Module, float, float]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = FloodWorldModel(
        dynamic_channels=int(checkpoint["dynamic_channels"]),
        static_channels=int(checkpoint["static_channels"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    discharge_stats = normalization["river_discharge"]

    discharge_mean = float(discharge_stats["mean"])
    discharge_std = max(float(discharge_stats["std"]), 1e-8)

    return model, discharge_mean, discharge_std


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    history: np.ndarray,
    static: np.ndarray,
) -> np.ndarray:
    x = torch.from_numpy(history[None])
    s = torch.from_numpy(static[None])

    output = model(x, s)

    if isinstance(output, (tuple, list)):
        output = output[0]

    prediction = (
        output
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if prediction.ndim != 2:
        raise RuntimeError(f"Unexpected model output shape: {prediction.shape}")

    if not np.isfinite(prediction).all():
        raise RuntimeError("Model produced NaN or Inf.")

    return prediction


def update_rainfall_features(
    current: np.ndarray,
    rainfall_raw: np.ndarray,
    rainfall_missing: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    previous_rainfall = denormalize(
        "precipitation",
        current[:, PRECIP],
        normalization,
    )

    rainfall_history = np.concatenate(
        [
            previous_rainfall,
            rainfall_raw[None],
        ],
        axis=0,
    )

    precip = rainfall_history[-1]
    precip_3d = rainfall_history[-3:].sum(axis=0)
    precip_7d = rainfall_history[-7:].sum(axis=0)
    precip_log1p = np.log1p(np.maximum(precip, 0.0))

    next_state = current[-1].copy()

    next_state[PRECIP] = normalize(
        "precipitation",
        precip,
        normalization,
    )

    next_state[PRECIP_3D] = normalize(
        "precip_3d",
        precip_3d,
        normalization,
    )

    next_state[PRECIP_7D] = normalize(
        "precip_7d",
        precip_7d,
        normalization,
    )

    next_state[PRECIP_LOG1P] = normalize(
        "precip_log1p",
        precip_log1p,
        normalization,
    )

    next_state[PRECIP_MISSING] = np.asarray(
        rainfall_missing,
        dtype=np.float32,
    )

    return next_state.astype(np.float32)


def load_future_rainfall(
    ds: xr.Dataset,
    start_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    precip = ds["precipitation"].isel(
        time=slice(
            start_index,
            start_index + horizon,
        )
    ).values.astype(np.float32)

    precip = np.nan_to_num(
        precip,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if "precip_missing" in ds:
        missing = ds["precip_missing"].isel(
            time=slice(
                start_index,
                start_index + horizon,
            )
        ).values.astype(np.float32)

        missing = np.nan_to_num(
            missing,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        missing = (missing > 0.5).astype(np.float32)
    else:
        missing = np.zeros_like(precip, dtype=np.float32)

    return precip, missing


def rollout_autoregressive(
    model: torch.nn.Module,
    initial_history: np.ndarray,
    static: np.ndarray,
    future_rainfall: np.ndarray,
    future_missing: np.ndarray,
    normalization: dict[str, Any],
    discharge_mean: float,
    discharge_std: float,
    mode: str,
) -> np.ndarray:
    current = initial_history.copy()
    predictions = []

    for day in range(HORIZON):
        print("-" * 80)
        print(f"ROLLOUT DAY {day + 1}/{HORIZON}")

        normalized_prediction = predict_one(
            model,
            current,
            static,
        )

        physical_prediction = (
            normalized_prediction * discharge_std
            + discharge_mean
        )

        physical_prediction = np.maximum(
            physical_prediction,
            0.0,
        ).astype(np.float32)

        predictions.append(physical_prediction)

        print(f"Predicted mean discharge: {physical_prediction.mean():.2f} m3/s")
        print(f"Predicted max discharge: {physical_prediction.max():.2f} m3/s")

        if mode == "oracle":
            rainfall_raw = future_rainfall[day]
            rainfall_missing = future_missing[day]

        elif mode == "persistent":
            rainfall_raw = denormalize(
                "precipitation",
                current[-1, PRECIP],
                normalization,
            )

            rainfall_raw = np.maximum(
                rainfall_raw,
                0.0,
            ).astype(np.float32)

            rainfall_missing = current[-1, PRECIP_MISSING].copy()

        else:
            raise ValueError(f"Unknown rollout mode: {mode}")

        next_state = update_rainfall_features(
            current=current,
            rainfall_raw=rainfall_raw,
            rainfall_missing=rainfall_missing,
            normalization=normalization,
        )

        next_state[DISCHARGE] = normalized_prediction

        current = np.concatenate(
            [
                current[1:],
                next_state[None],
            ],
            axis=0,
        )

    return np.stack(predictions, axis=0).astype(np.float32)


def rollout_teacher_forced(
    model: torch.nn.Module,
    ds: xr.Dataset,
    static: np.ndarray,
    forecast_start: int,
    normalization: dict[str, Any],
    discharge_mean: float,
    discharge_std: float,
) -> np.ndarray:
    predictions = []

    for day in range(HORIZON):
        target_index = forecast_start + day

        history = load_history(
            ds,
            target_index,
            normalization,
        )

        normalized_prediction = predict_one(
            model,
            history,
            static,
        )

        physical_prediction = (
            normalized_prediction * discharge_std
            + discharge_mean
        )

        physical_prediction = np.maximum(
            physical_prediction,
            0.0,
        ).astype(np.float32)

        predictions.append(physical_prediction)

    return np.stack(predictions, axis=0).astype(np.float32)


def evaluate_day(
    prediction: np.ndarray,
    actual: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    mask = (
        (valid_mask > 0.5)
        & np.isfinite(actual)
        & np.isfinite(prediction)
    )

    valid_cells = int(mask.sum())

    if valid_cells == 0:
        return {
            "valid_cells": 0,
            "valid_fraction": 0.0,
            "mae_m3_s": None,
            "rmse_m3_s": None,
            "bias_m3_s": None,
            "correlation": None,
        }

    y = actual[mask].astype(np.float64)
    p = prediction[mask].astype(np.float64)

    error = p - y

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0:
        correlation = float(np.corrcoef(y, p)[0, 1])
    else:
        correlation = None

    return {
        "valid_cells": valid_cells,
        "valid_fraction": float(mask.mean()),
        "mae_m3_s": mae,
        "rmse_m3_s": rmse,
        "bias_m3_s": bias,
        "correlation": correlation,
    }


def evaluate_forecast(
    predictions: np.ndarray,
    actuals: np.ndarray,
    valid_masks: np.ndarray,
) -> list[dict[str, Any]]:
    metrics = []

    for day in range(HORIZON):
        result = evaluate_day(
            predictions[day],
            actuals[day],
            valid_masks[day],
        )

        result["lead_day"] = day + 1
        metrics.append(result)

    return metrics


def calculate_skill(
    model_rmse: float | None,
    baseline_rmse: float | None,
) -> float | None:
    if (
        model_rmse is None
        or baseline_rmse is None
        or baseline_rmse == 0
    ):
        return None

    return float(1.0 - model_rmse / baseline_rmse)


def build_persistence_forecast(
    initial_current_discharge: np.ndarray,
) -> np.ndarray:
    return np.repeat(
        initial_current_discharge[None, ...],
        HORIZON,
        axis=0,
    ).astype(np.float32)


def main() -> None:
    os.chdir(PROJECT_ROOT)

    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL V0")
    print("7-DAY ROLLOUT DIAGNOSTIC")
    print("=" * 80)

    check_files()

    normalization = load_normalization()

    model, discharge_mean, discharge_std = build_model(
        normalization
    )

    static, river_mask, lat, lon = load_static(
        normalization
    )

    ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    total_time = ds.sizes["time"]

    if total_time < HISTORY_LENGTH + HORIZON:
        ds.close()
        raise RuntimeError("Not enough temporal data for rollout.")

    forecast_start = total_time - HORIZON

    initial_history = load_history(
        ds,
        forecast_start,
        normalization,
    )

    history_dates = ds.time.values[
        forecast_start - HISTORY_LENGTH:
        forecast_start
    ]

    future_dates = ds.time.values[
        forecast_start:
        forecast_start + HORIZON
    ]

    print(f"State date: {history_dates[-1]}")
    print(f"Forecast start: {future_dates[0]}")
    print(f"Forecast end: {future_dates[-1]}")
    print(f"Dynamic shape: {initial_history.shape}")
    print(f"Static shape: {static.shape}")
    print(f"River-mask cells: {int(river_mask.sum())}")

    future_rainfall, future_missing = load_future_rainfall(
        ds,
        forecast_start,
        HORIZON,
    )

    print()
    print("=" * 80)
    print("RUNNING TEACHER-FORCED FORECAST")
    print("=" * 80)

    teacher_forced = rollout_teacher_forced(
        model=model,
        ds=ds,
        static=static,
        forecast_start=forecast_start,
        normalization=normalization,
        discharge_mean=discharge_mean,
        discharge_std=discharge_std,
    )

    print()
    print("=" * 80)
    print("RUNNING AUTOREGRESSIVE + ORACLE RAINFALL")
    print("=" * 80)

    ar_oracle = rollout_autoregressive(
        model=model,
        initial_history=initial_history,
        static=static,
        future_rainfall=future_rainfall,
        future_missing=future_missing,
        normalization=normalization,
        discharge_mean=discharge_mean,
        discharge_std=discharge_std,
        mode="oracle",
    )

    print()
    print("=" * 80)
    print("RUNNING AUTOREGRESSIVE + PERSISTENT RAINFALL")
    print("=" * 80)

    ar_persistent = rollout_autoregressive(
        model=model,
        initial_history=initial_history,
        static=static,
        future_rainfall=future_rainfall,
        future_missing=future_missing,
        normalization=normalization,
        discharge_mean=discharge_mean,
        discharge_std=discharge_std,
        mode="persistent",
    )

    actuals = []
    valid_masks = []

    for day in range(HORIZON):
        idx = forecast_start + day

        actual = ds["river_discharge"].isel(
            time=idx
        ).values.astype(np.float32)

        glofas_valid = ds["glofas_discharge_valid_t"].isel(
            time=idx
        ).values.astype(np.float32)

        evaluation_mask = (
            (glofas_valid > 0.5)
            & (river_mask > 0.5)
        ).astype(np.float32)

        actuals.append(actual)
        valid_masks.append(evaluation_mask)

    actuals = np.stack(
        actuals,
        axis=0,
    ).astype(np.float32)

    valid_masks = np.stack(
        valid_masks,
        axis=0,
    ).astype(np.float32)

    current_discharge_physical = denormalize(
        "river_discharge",
        initial_history[-1, DISCHARGE],
        normalization,
    )

    current_discharge_physical = np.maximum(
        current_discharge_physical,
        0.0,
    ).astype(np.float32)

    persistence = build_persistence_forecast(
        current_discharge_physical
    )

    all_predictions = {
        "teacher_forced": teacher_forced,
        "autoregressive_oracle": ar_oracle,
        "autoregressive_persistent": ar_persistent,
        "persistence": persistence,
    }

    all_metrics: dict[str, Any] = {
        "metadata": {
            "model": "World Model V0",
            "checkpoint": str(CHECKPOINT_PATH),
            "dynamic_dataset": str(DYNAMIC_PATH),
            "static_dataset": str(STATIC_PATH),
            "normalization": str(NORMALIZATION_PATH),
            "history_length": HISTORY_LENGTH,
            "horizon_days": HORIZON,
            "forecast_start": str(future_dates[0]),
            "forecast_end": str(future_dates[-1]),
            "evaluation_mask": "glofas_discharge_valid_t × river_mask",
        },
        "experiments": {},
    }

    print()
    print("=" * 80)
    print("7-DAY RESULTS")
    print("=" * 80)

    persistence_metrics = evaluate_forecast(
        persistence,
        actuals,
        valid_masks,
    )

    all_metrics["experiments"]["persistence"] = persistence_metrics

    for day_metrics in persistence_metrics:
        print("-" * 80)
        print(f"PERSISTENCE DAY {day_metrics['lead_day']}")
        print(f"MAE: {day_metrics['mae_m3_s']}")
        print(f"RMSE: {day_metrics['rmse_m3_s']}")
        print(f"Bias: {day_metrics['bias_m3_s']}")
        print(f"Correlation: {day_metrics['correlation']}")

    for name in [
        "teacher_forced",
        "autoregressive_oracle",
        "autoregressive_persistent",
    ]:
        print()
        print("=" * 80)
        print(name.upper())
        print("=" * 80)

        metrics = evaluate_forecast(
            all_predictions[name],
            actuals,
            valid_masks,
        )

        for day_metrics in metrics:
            lead = day_metrics["lead_day"]

            baseline_rmse = persistence_metrics[
                lead - 1
            ]["rmse_m3_s"]

            skill = calculate_skill(
                day_metrics["rmse_m3_s"],
                baseline_rmse,
            )

            day_metrics["skill_vs_persistence"] = skill

            print("-" * 80)
            print(f"DAY {lead}")
            print(f"Valid cells: {day_metrics['valid_cells']}")
            print(f"Valid fraction: {day_metrics['valid_fraction']}")
            print(f"MAE: {day_metrics['mae_m3_s']}")
            print(f"RMSE: {day_metrics['rmse_m3_s']}")
            print(f"Bias: {day_metrics['bias_m3_s']}")
            print(f"Correlation: {day_metrics['correlation']}")
            print(f"Skill vs persistence: {skill}")

        all_metrics["experiments"][name] = metrics

    prediction_stack = np.stack(
        [
            all_predictions["teacher_forced"],
            all_predictions["autoregressive_oracle"],
            all_predictions["autoregressive_persistent"],
            all_predictions["persistence"],
        ],
        axis=0,
    )

    FORECAST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_ds = xr.Dataset(
        {
            "predicted_river_discharge": (
                (
                    "experiment",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                prediction_stack,
            ),
            "actual_river_discharge": (
                (
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                actuals,
            ),
            "evaluation_mask": (
                (
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                valid_masks,
            ),
            "future_precipitation": (
                (
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                future_rainfall,
            ),
        },
        coords={
            "experiment": [
                "teacher_forced",
                "autoregressive_oracle",
                "autoregressive_persistent",
                "persistence",
            ],
            "forecast_day": np.arange(
                1,
                HORIZON + 1,
            ),
            "lat": lat,
            "lon": lon,
        },
    )

    forecast_ds["predicted_river_discharge"].attrs["units"] = "m3 s-1"
    forecast_ds["actual_river_discharge"].attrs["units"] = "m3 s-1"
    forecast_ds["future_precipitation"].attrs["units"] = "original_dataset_units"
    forecast_ds.attrs["model"] = "World Model V0"
    forecast_ds.attrs["forecast_type"] = "7-day diagnostic rollout"
    forecast_ds.attrs["evaluation_mask"] = "glofas_discharge_valid_t × river_mask"

    forecast_ds.to_netcdf(
        FORECAST_PATH
    )

    forecast_ds.close()

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=2,
        )

    ds.close()

    print()
    print("=" * 80)
    print("ROLLOUT COMPLETE")
    print("=" * 80)
    print(f"Forecast: {FORECAST_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print()
    print("Experiments:")
    print("  teacher_forced")
    print("  autoregressive_oracle")
    print("  autoregressive_persistent")
    print("  persistence")


if __name__ == "__main__":
    main()