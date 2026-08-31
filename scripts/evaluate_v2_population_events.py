from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT / "src"
    ),
)

from flood_world_model.datasets.v2_population import (
    V2PopulationDataset,
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
    / "data/features/training_v3/v2_population_normalization.json"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/v2_vs_population_test.nc"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/v2_population_event_analysis.json"
)

HISTORY_LENGTH = 14
HORIZON = 7

TOP_RIVER_CELLS = 25

EVENT_PERCENTILE = 90.0
MIN_EVENT_GAP = 2

EVENT_WINDOW_BEFORE = 2
EVENT_WINDOW_AFTER = 4

SEVERE_UNDERPREDICTION_RATIO = 0.90


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def denormalize_discharge(
    values: np.ndarray,
    normalization: dict,
) -> np.ndarray:
    stats = normalization[
        "river_discharge"
    ]

    mean = float(
        stats["mean"]
    )

    std = max(
        float(stats["std"]),
        1e-8,
    )

    return (
        values * std + mean
    ).astype(np.float32)


def detect_events(
    series: np.ndarray,
    percentile: float,
    min_gap: int,
) -> list[int]:
    finite = np.isfinite(
        series
    )

    if not finite.any():
        return []

    threshold = float(
        np.percentile(
            series[finite],
            percentile,
        )
    )

    candidates = np.where(
        finite
        & (
            series >= threshold
        )
    )[0]

    if len(candidates) == 0:
        return []

    events: list[int] = []

    for index in candidates:

        index = int(
            index
        )

        if len(events) == 0:
            events.append(
                index
            )
            continue

        previous = events[-1]

        if (
            index - previous
            > min_gap
        ):
            events.append(
                index
            )

        elif (
            series[index]
            > series[previous]
        ):
            events[-1] = index

    return events


def safe_relative_error(
    predicted: float,
    actual: float,
) -> float:
    if actual == 0.0:
        return float("nan")

    return float(
        abs(
            predicted - actual
        )
        / abs(actual)
    )


def main() -> None:
    print("=" * 80)
    print(
        "BANGLADESH FLOOD WORLD MODEL"
    )
    print(
        "V2-POPULATION HIGH-FLOW SPATIAL EVENT ANALYSIS"
    )
    print("=" * 80)

    required_files = [
        DYNAMIC_PATH,
        STATIC_PATH,
        NORMALIZATION_PATH,
        STATIC_PATH,
        FORECAST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    normalization = load_json(
        NORMALIZATION_PATH
    )

    # -------------------------------------------------------------------------
    # Load the original dynamic dataset.
    # -------------------------------------------------------------------------

    ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    total_time = ds.sizes[
        "time"
    ]

    train_end = int(
        total_time * 0.70
    )

    validation_end = int(
        total_time * 0.85
    )

    test_end = total_time

    lat = ds.lat.values
    lon = ds.lon.values

    # Use the same raw observations for event detection.
    all_discharge = (
        ds[
            "river_discharge"
        ]
        .values
        .astype(np.float32)
    )

    all_valid = (
        ds[
            "glofas_discharge_valid_t"
        ]
        .values
        .astype(np.float32)
    )

    time_values = (
        ds.time.values
    )

    ds.close()

    # -------------------------------------------------------------------------
    # Load the static river mask.
    # -------------------------------------------------------------------------

    static_ds = xr.open_zarr(
        STATIC_PATH,
        consolidated=True,
    )

    river_mask = (
        static_ds[
            "river_mask"
        ]
        .values
        .astype(np.float32)
    )

    river_mask = (
        river_mask > 0.5
    )

    static_ds.close()

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Build the exact same V2-Population test dataset used for training/eval.
    # This gives us the actual target_start indices for every saved sample.
    # We must NOT assume:
    #
    #     number_of_samples == number_of_test_days
    #
    # because each sample represents a 7-day forecast window.
    # -------------------------------------------------------------------------

    test_dataset = V2PopulationDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=validation_end,
        end_index=test_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    sample_indices = (
        test_dataset.indices.copy()
    )

    sample_count = len(
        sample_indices
    )

    print(
        f"Test forecast windows: {sample_count}"
    )

    # -------------------------------------------------------------------------
    # Load saved V2-Population predictions.
    # -------------------------------------------------------------------------

    forecast_ds = xr.open_dataset(
        FORECAST_PATH
    )

    prediction = (
        forecast_ds[
            "v2_population_predicted_discharge"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    saved_actual = (
        forecast_ds[
            "actual_discharge"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    saved_mask = (
        forecast_ds[
            "evaluation_mask"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    forecast_lat = (
        forecast_ds.lat.values
    )

    forecast_lon = (
        forecast_ds.lon.values
    )

    forecast_ds.close()

    # -------------------------------------------------------------------------
    # Verify the saved forecast shape.
    # -------------------------------------------------------------------------

    expected_shape = (
        sample_count,
        len(forecast_lat),
        len(forecast_lon),
    )

    if prediction.shape != expected_shape:
        raise RuntimeError(
            "Saved forecast sample count does not match "
            "the V2-Population test dataset."
        )

    if saved_actual.shape != prediction.shape:
        raise RuntimeError(
            "Saved actual shape does not match prediction shape."
        )

    if saved_mask.shape != prediction.shape:
        raise RuntimeError(
            "Saved mask shape does not match prediction shape."
        )

    if (
        not np.array_equal(
            forecast_lat,
            lat,
        )
        or not np.array_equal(
            forecast_lon,
            lon,
        )
    ):
        raise RuntimeError(
            "Forecast and dynamic dataset coordinates do not match."
        )

    print(
        f"Prediction shape: {prediction.shape}"
    )

    # -------------------------------------------------------------------------
    # The comparison NetCDF stores physical-unit predictions.
    #
    # Detect whether the values appear already physical.
    # If values look normalized, denormalize them.
    # -------------------------------------------------------------------------

    p95 = float(
        np.nanpercentile(
            np.abs(prediction),
            95,
        )
    )

    if p95 < 100.0:
        prediction = denormalize_discharge(
            prediction,
            normalization,
        )

    prediction = np.maximum(
        prediction,
        0.0,
    )

    # -------------------------------------------------------------------------
    # Construct the corresponding actual observations from the original Zarr.
    #
    # sample_indices are target_start indices generated by the dataset.
    # forecast_day=1 corresponds exactly to target_start.
    # -------------------------------------------------------------------------

    actual_series = []
    valid_series = []
    sample_dates = []

    for sample_index in sample_indices:

        sample_index = int(
            sample_index
        )

        actual = (
            all_discharge[
                sample_index
            ]
        )

        valid = (
            (
                all_valid[
                    sample_index
                ]
                > 0.5
            )
            & river_mask
        )

        actual_series.append(
            actual
        )

        valid_series.append(
            valid.astype(
                np.float32
            )
        )

        sample_dates.append(
            time_values[
                sample_index
            ]
        )

    actual_series = np.stack(
        actual_series,
        axis=0,
    ).astype(np.float32)

    valid_series = np.stack(
        valid_series,
        axis=0,
    ).astype(np.float32)

    # -------------------------------------------------------------------------
    # Compare saved actuals against source observations.
    # -------------------------------------------------------------------------

    source_actual_masked = np.where(
        valid_series > 0.5,
        actual_series,
        np.nan,
    )

    saved_actual_masked = np.where(
        saved_mask > 0.5,
        saved_actual,
        np.nan,
    )

    comparable = (
        np.isfinite(
            source_actual_masked
        )
        & np.isfinite(
            saved_actual_masked
        )
    )

    if comparable.any():

        max_actual_difference = float(
            np.nanmax(
                np.abs(
                    source_actual_masked[
                        comparable
                    ]
                    - saved_actual_masked[
                        comparable
                    ]
                )
            )
        )

        print(
            f"Maximum saved/source actual difference: {max_actual_difference:.6f} m3/s"
        )

    # -------------------------------------------------------------------------
    # Apply authoritative mask.
    # -------------------------------------------------------------------------

    actual_series = np.where(
        valid_series > 0.5,
        actual_series,
        np.nan,
    )

    prediction = np.where(
        valid_series > 0.5,
        prediction,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Select top river cells using observed high-flow magnitude.
    # -------------------------------------------------------------------------

    cell_peaks = np.nanmax(
        actual_series,
        axis=0,
    )

    river_indices = np.where(
        river_mask
    )

    peak_values = cell_peaks[
        river_indices
    ]

    finite_order = np.argsort(
        np.where(
            np.isfinite(
                peak_values
            ),
            peak_values,
            -np.inf,
        )
    )[::-1]

    selected_count = min(
        TOP_RIVER_CELLS,
        len(finite_order),
    )

    selected_cells = []

    for rank in range(
        selected_count
    ):

        position = int(
            finite_order[
                rank
            ]
        )

        lat_index = int(
            river_indices[
                0
            ][
                position
            ]
        )

        lon_index = int(
            river_indices[
                1
            ][
                position
            ]
        )

        selected_cells.append(
            (
                lat_index,
                lon_index,
            )
        )

    print(
        f"Selected river cells: {selected_count}"
    )

    # -------------------------------------------------------------------------
    # Detect and evaluate events.
    # -------------------------------------------------------------------------

    events = []

    relative_peak_errors = []
    timing_errors = []

    severe_underprediction_count = 0

    for lat_index, lon_index in selected_cells:

        observed = (
            actual_series[
                :,
                lat_index,
                lon_index,
            ]
        )

        predicted = (
            prediction[
                :,
                lat_index,
                lon_index,
            ]
        )

        if not np.isfinite(
            observed
        ).any():
            continue

        detected_events = detect_events(
            observed,
            EVENT_PERCENTILE,
            MIN_EVENT_GAP,
        )

        for event_number, event_index in enumerate(
            detected_events,
            start=1,
        ):

            if not np.isfinite(
                observed[
                    event_index
                ]
            ):
                continue

            window_start = max(
                0,
                event_index
                - EVENT_WINDOW_BEFORE,
            )

            window_end = min(
                len(predicted),
                event_index
                + EVENT_WINDOW_AFTER
                + 1,
            )

            predicted_window = (
                predicted[
                    window_start:
                    window_end
                ]
            )

            if not np.isfinite(
                predicted_window
            ).any():
                continue

            predicted_peak_relative = int(
                np.nanargmax(
                    predicted_window
                )
            )

            predicted_peak_index = (
                window_start
                + predicted_peak_relative
            )

            actual_peak = float(
                observed[
                    event_index
                ]
            )

            predicted_peak = float(
                predicted[
                    predicted_peak_index
                ]
            )

            relative_error = (
                safe_relative_error(
                    predicted_peak,
                    actual_peak,
                )
            )

            timing_error = float(
                abs(
                    predicted_peak_index
                    - event_index
                )
            )

            severe_underprediction = (
                predicted_peak
                < (
                    SEVERE_UNDERPREDICTION_RATIO
                    * actual_peak
                )
            )

            if severe_underprediction:
                severe_underprediction_count += 1

            relative_peak_errors.append(
                relative_error
            )

            timing_errors.append(
                timing_error
            )

            events.append(
                {
                    "sample_index": int(
                        sample_indices[
                            event_index
                        ]
                    )
                    if event_index
                    < len(sample_indices)
                    else None,
                    "lat_index": lat_index,
                    "lon_index": lon_index,
                    "event_number_for_cell": event_number,
                    "event_sample_position": int(
                        event_index
                    ),
                    "event_date": str(
                        sample_dates[
                            event_index
                        ]
                    ),
                    "actual_peak_m3_s": actual_peak,
                    "predicted_peak_m3_s": predicted_peak,
                    "relative_peak_error": float(
                        relative_error
                    ),
                    "relative_peak_error_percent": float(
                        relative_error
                        * 100.0
                    ),
                    "peak_timing_error_days": timing_error,
                    "severe_underprediction": bool(
                        severe_underprediction
                    ),
                }
            )

    relative_peak_errors = np.asarray(
        relative_peak_errors,
        dtype=np.float64,
    )

    timing_errors = np.asarray(
        timing_errors,
        dtype=np.float64,
    )

    total_events = len(
        events
    )

    if total_events == 0:
        raise RuntimeError(
            "No valid high-flow events detected."
        )

    mean_relative_peak_error = float(
        np.nanmean(
            relative_peak_errors
        )
    )

    median_relative_peak_error = float(
        np.nanmedian(
            relative_peak_errors
        )
    )

    mean_timing_error = float(
        np.mean(
            timing_errors
        )
    )

    median_timing_error = float(
        np.median(
            timing_errors
        )
    )

    severe_rate = float(
        severe_underprediction_count
        / total_events
    )

    # -------------------------------------------------------------------------
    # Output.
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("V2-POPULATION EVENT RESULTS")
    print("=" * 80)

    print(
        f"Forecast windows: {sample_count}"
    )

    print(
        f"Test forecast start: {sample_dates[0]}"
    )

    print(
        f"Test forecast end: {sample_dates[-1]}"
    )

    print(
        f"River cells analyzed: {selected_count}"
    )

    print(
        f"Total high-flow events: {total_events}"
    )

    print(
        f"Mean relative peak error: {mean_relative_peak_error * 100.0:.2f}%"
    )

    print(
        f"Median relative peak error: {median_relative_peak_error * 100.0:.2f}%"
    )

    print(
        f"Mean peak timing error: {mean_timing_error:.2f} days"
    )

    print(
        f"Median peak timing error: {median_timing_error:.2f} days"
    )

    print(
        f"Severe underprediction count: {severe_underprediction_count}"
    )

    print(
        f"Severe underprediction rate: {severe_rate * 100.0:.2f}%"
    )

    results = {
        "model": "Bangladesh Flood World Model V2-Population",
        "evaluation_type": "high_flow_spatial_event_analysis",
        "forecast_used": "forecast_day_1",
        "forecast_windows": int(
            sample_count
        ),
        "test_forecast_start_date": str(
            sample_dates[0]
        ),
        "test_forecast_end_date": str(
            sample_dates[-1]
        ),
        "top_river_cells": int(
            selected_count
        ),
        "event_percentile": EVENT_PERCENTILE,
        "minimum_event_gap_days": MIN_EVENT_GAP,
        "event_window_before_days": EVENT_WINDOW_BEFORE,
        "event_window_after_days": EVENT_WINDOW_AFTER,
        "severe_underprediction_threshold": (
            SEVERE_UNDERPREDICTION_RATIO
        ),
        "total_events": int(
            total_events
        ),
        "mean_relative_peak_error": (
            mean_relative_peak_error
        ),
        "mean_relative_peak_error_percent": (
            mean_relative_peak_error
            * 100.0
        ),
        "median_relative_peak_error": (
            median_relative_peak_error
        ),
        "median_relative_peak_error_percent": (
            median_relative_peak_error
            * 100.0
        ),
        "mean_peak_timing_error_days": (
            mean_timing_error
        ),
        "median_peak_timing_error_days": (
            median_timing_error
        ),
        "severe_underprediction_count": (
            int(
                severe_underprediction_count
            )
        ),
        "severe_underprediction_rate": (
            severe_rate
        ),
        "severe_underprediction_rate_percent": (
            severe_rate
            * 100.0
        ),
        "population_feature": True,
        "population_source": (
            "data/features/static_v3.zarr/population_density"
        ),
        "evaluation_mask": (
            "glofas_discharge_valid_t × river_mask"
        ),
        "forecast_file": str(
            FORECAST_PATH
        ),
        "events": events,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
