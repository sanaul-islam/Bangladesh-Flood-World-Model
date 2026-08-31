from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


from flood_world_model.models.world_model import (
    FloodWorldModel,
)

from flood_world_model.datasets.multihorizon import (
    MultiHorizonFloodDataset,
)


DYNAMIC_PATH = (
    PROJECT_ROOT
    / "data/features/dynamic_core_v2.zarr"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "data/features/static_v3.zarr"
)

V0_NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/features/training_v3/normalization.json"
)

V2_NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/features/training_v3/v2_normalization.json"
)

V2_FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/v2_vs_population_test.nc"
)

V0_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/world_model_v0_best.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/canonical_event_comparison.json"
)


HISTORY_LENGTH = 14
HORIZON = 7

TOP_RIVER_CELLS = 25

EVENT_PERCENTILE = 95.0
MIN_EVENT_GAP_DAYS = 7

PEAK_MATCH_WINDOW_BEFORE = 3
PEAK_MATCH_WINDOW_AFTER = 3

SEVERE_UNDERPREDICTION_RATIO = 0.90

BATCH_SIZE = 1


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


def get_split_indices(
    total_time: int,
) -> tuple[int, int, int]:
    train_end = int(
        total_time * 0.70
    )

    validation_end = int(
        total_time * 0.85
    )

    return (
        train_end,
        validation_end,
        total_time,
    )


def detect_events(
    series: np.ndarray,
    percentile: float,
    minimum_gap_days: int,
) -> list[int]:
    finite = np.isfinite(
        series
    )

    if finite.sum() == 0:
        return []

    threshold = float(
        np.nanpercentile(
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

    events = []

    current_peak = None
    current_peak_value = -np.inf
    last_candidate = None

    for raw_index in candidates:
        index = int(
            raw_index
        )

        value = float(
            series[index]
        )

        if last_candidate is None:
            current_peak = index
            current_peak_value = value
            last_candidate = index
            continue

        gap = (
            index
            - last_candidate
        )

        if gap <= minimum_gap_days:
            if value > current_peak_value:
                current_peak = index
                current_peak_value = value
        else:
            if current_peak is not None:
                events.append(
                    current_peak
                )

            current_peak = index
            current_peak_value = value

        last_candidate = index

    if current_peak is not None:
        events.append(
            current_peak
        )

    return events


def select_top_river_cells(
    actuals: np.ndarray,
    masks: np.ndarray,
    river_mask: np.ndarray,
) -> list[tuple[int, int]]:
    masked_actual = np.where(
        masks > 0.5,
        actuals,
        np.nan,
    )

    valid_cells = (
        np.isfinite(
            masked_actual
        ).any(axis=0)
        & (
            river_mask > 0.5
        )
    )

    if not valid_cells.any():
        raise RuntimeError(
            "No valid river cells available."
        )

    rows, cols = np.where(
        valid_cells
    )

    cell_peaks = []

    for row, col in zip(
        rows,
        cols,
    ):
        values = masked_actual[
            :,
            row,
            col,
        ]

        finite = np.isfinite(
            values
        )

        if finite.any():
            cell_peaks.append(
                (
                    int(row),
                    int(col),
                    float(
                        np.max(
                            values[
                                finite
                            ]
                        )
                    ),
                )
            )

    if not cell_peaks:
        raise RuntimeError(
            "No finite river-cell peaks available."
        )

    cell_peaks.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    selected = [
        (
            item[0],
            item[1],
        )
        for item in cell_peaks[
            :TOP_RIVER_CELLS
        ]
    ]

    return selected


@torch.inference_mode()
def generate_v0_predictions(
    dataset: MultiHorizonFloodDataset,
    normalization: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    print("=" * 80)
    print("GENERATING V0 PREDICTIONS")
    print("=" * 80)

    checkpoint = torch.load(
        V0_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = FloodWorldModel(
        dynamic_channels=int(
            checkpoint[
                "dynamic_channels"
            ]
        ),
        static_channels=int(
            checkpoint[
                "static_channels"
            ]
        ),
        hidden_channels=int(
            checkpoint[
                "hidden_channels"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    predictions = []

    total_batches = len(
        loader
    )

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):
        output = model(
            batch["history"],
            batch["static"],
        )

        if isinstance(
            output,
            (tuple, list),
        ):
            output = output[0]

        prediction = (
            output[
                :, 0
            ]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        predictions.append(
            prediction
        )

        if (
            batch_number % 100 == 0
            or batch_number == total_batches
        ):
            print(
                f"V0 samples: {batch_number}/{total_batches}"
            )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    predictions = denormalize_discharge(
        predictions,
        normalization,
    )

    predictions = np.maximum(
        predictions,
        0.0,
    )

    sample_indices = (
        dataset.indices.copy()
    )

    if len(
        sample_indices
    ) != predictions.shape[0]:
        raise RuntimeError(
            "V0 prediction count does not match dataset sample indices."
        )

    return (
        predictions,
        sample_indices,
    )


def load_saved_v2_predictions(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ds = xr.open_dataset(
        path
    )

    v2 = (
        ds[
            "v2_predicted_discharge"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    v2_population = (
        ds[
            "v2_population_predicted_discharge"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    actual = (
        ds[
            "actual_discharge"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    mask = (
        ds[
            "evaluation_mask"
        ]
        .isel(
            forecast_day=0
        )
        .values
        .astype(np.float32)
    )

    lat = ds.lat.values
    lon = ds.lon.values

    ds.close()

    return (
        v2,
        v2_population,
        actual,
        mask,
    )


def evaluate_model_events(
    model_name: str,
    prediction_grid: np.ndarray,
    actuals: np.ndarray,
    masks: np.ndarray,
    dates: np.ndarray,
    selected_cells: list[tuple[int, int]],
) -> dict:
    relative_errors = []
    timing_errors = []

    severe_count = 0

    event_records = []

    for cell_number, (
        row,
        col,
    ) in enumerate(
        selected_cells,
        start=1,
    ):
        actual_series = (
            actuals[
                :,
                row,
                col,
            ]
        )

        prediction_series = (
            prediction_grid[
                :,
                row,
                col,
            ]
        )

        if not np.isfinite(
            actual_series
        ).any():
            continue

        events = detect_events(
            actual_series,
            EVENT_PERCENTILE,
            MIN_EVENT_GAP_DAYS,
        )

        for event_index in events:
            actual_peak = float(
                actual_series[
                    event_index
                ]
            )

            if not np.isfinite(
                actual_peak
            ):
                continue

            start = max(
                0,
                event_index
                - PEAK_MATCH_WINDOW_BEFORE,
            )

            end = min(
                len(
                    prediction_series
                ),
                event_index
                + PEAK_MATCH_WINDOW_AFTER
                + 1,
            )

            prediction_window = (
                prediction_series[
                    start:end
                ]
            )

            finite = np.isfinite(
                prediction_window
            )

            if not finite.any():
                continue

            positions = np.where(
                finite
            )[0]

            values = (
                prediction_window[
                    finite
                ]
            )

            best_position = int(
                positions[
                    np.argmax(
                        values
                    )
                ]
            )

            predicted_index = (
                start
                + best_position
            )

            predicted_peak = float(
                prediction_series[
                    predicted_index
                ]
            )

            relative_error = float(
                abs(
                    predicted_peak
                    - actual_peak
                )
                / max(
                    abs(actual_peak),
                    1e-8,
                )
            )

            timing_error = float(
                abs(
                    predicted_index
                    - event_index
                )
            )

            severe = (
                predicted_peak
                < (
                    SEVERE_UNDERPREDICTION_RATIO
                    * actual_peak
                )
            )

            if severe:
                severe_count += 1

            relative_errors.append(
                relative_error
            )

            timing_errors.append(
                timing_error
            )

            event_records.append(
                {
                    "model": model_name,
                    "cell_number": cell_number,
                    "lat_index": row,
                    "lon_index": col,
                    "event_index": int(
                        event_index
                    ),
                    "event_date": str(
                        dates[
                            event_index
                        ]
                    ),
                    "actual_peak_m3_s": actual_peak,
                    "predicted_peak_m3_s": predicted_peak,
                    "relative_peak_error_percent": float(
                        relative_error
                        * 100.0
                    ),
                    "timing_error_days": timing_error,
                    "predicted_peak_date": str(
                        dates[
                            predicted_index
                        ]
                    ),
                    "severe_underprediction": bool(
                        severe
                    ),
                }
            )

    if not relative_errors:
        return {
            "model": model_name,
            "total_events": 0,
            "mean_relative_peak_error_percent": None,
            "median_relative_peak_error_percent": None,
            "mean_peak_timing_error_days": None,
            "median_peak_timing_error_days": None,
            "severe_underprediction_count": 0,
            "severe_underprediction_rate_percent": None,
            "events": [],
        }

    return {
        "model": model_name,
        "total_events": len(
            relative_errors
        ),
        "mean_relative_peak_error_percent": float(
            np.mean(
                relative_errors
            )
        ),
        "median_relative_peak_error_percent": float(
            np.median(
                relative_errors
            )
        ),
        "mean_peak_timing_error_days": float(
            np.mean(
                timing_errors
            )
        ),
        "median_peak_timing_error_days": float(
            np.median(
                timing_errors
            )
        ),
        "severe_underprediction_count": int(
            severe_count
        ),
        "severe_underprediction_rate_percent": float(
            severe_count
            / len(relative_errors)
            * 100.0
        ),
        "events": event_records,
    }


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("CANONICAL V0 vs V2 vs V2-POPULATION EVENT ANALYSIS")
    print("=" * 80)

    required_files = [
        DYNAMIC_PATH,
        STATIC_PATH,
        V0_NORMALIZATION_PATH,
        V0_CHECKPOINT_PATH,
        V2_FORECAST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    v0_normalization = load_json(
        V0_NORMALIZATION_PATH
    )

    ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    total_time = ds.sizes[
        "time"
    ]

    _, test_start, test_end = (
        get_split_indices(
            total_time
        )
    )

    lat = ds.lat.values
    lon = ds.lon.values

    time_values = ds.time.values

    discharge_all = (
        ds[
            "river_discharge"
        ]
        .values
        .astype(np.float32)
    )

    valid_all = (
        ds[
            "glofas_discharge_valid_t"
        ]
        .values
        .astype(np.float32)
    )

    ds.close()

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
    ).astype(np.float32)

    static_ds.close()

    print(
        f"Test range: {test_start} -> {test_end}"
    )

    # -------------------------------------------------------------------------
    # Construct the exact 7-day test-window index set used by V2.
    # -------------------------------------------------------------------------

    common_dataset = MultiHorizonFloodDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=V2_NORMALIZATION_PATH,
        start_index=test_start,
        end_index=test_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    sample_indices = (
        common_dataset.indices.copy()
    )

    sample_count = len(
        sample_indices
    )

    print(
        f"Canonical test windows: {sample_count}"
    )

    # -------------------------------------------------------------------------
    # Load V2 and V2-Population physical-unit predictions.
    # -------------------------------------------------------------------------

    (
        v2_predictions,
        v2_population_predictions,
        saved_actual,
        saved_mask,
    ) = load_saved_v2_predictions(
        V2_FORECAST_PATH
    )

    if v2_predictions.shape != (
        sample_count,
        60,
        45,
    ):
        raise RuntimeError(
            f"Unexpected V2 prediction shape: {v2_predictions.shape}"
        )

    if v2_population_predictions.shape != (
        sample_count,
        60,
        45,
    ):
        raise RuntimeError(
            f"Unexpected V2-Population prediction shape: {v2_population_predictions.shape}"
        )

    # -------------------------------------------------------------------------
    # Create common daily observations.
    # -------------------------------------------------------------------------

    actuals = []
    masks = []
    dates = []

    for target_index in sample_indices:

        target_index = int(
            target_index
        )

        actual = (
            discharge_all[
                target_index
            ]
        )

        mask = (
            (
                valid_all[
                    target_index
                ]
                > 0.5
            )
            & (
                river_mask > 0.5
            )
        ).astype(
            np.float32
        )

        actuals.append(
            actual
        )

        masks.append(
            mask
        )

        dates.append(
            time_values[
                target_index
            ]
        )

    actuals = np.stack(
        actuals,
        axis=0,
    ).astype(np.float32)

    masks = np.stack(
        masks,
        axis=0,
    ).astype(np.float32)

    dates = np.asarray(
        dates
    )

    # -------------------------------------------------------------------------
    # Verify saved targets and masks.
    # -------------------------------------------------------------------------

    comparable = (
        np.isfinite(
            actuals
        )
        & np.isfinite(
            saved_actual
        )
    )

    if comparable.any():
        maximum_difference = float(
            np.max(
                np.abs(
                    actuals[
                        comparable
                    ]
                    - saved_actual[
                        comparable
                    ]
                )
            )
        )

        print(
            f"Maximum source/saved actual difference: {maximum_difference:.6f} m3/s"
        )

    if not np.array_equal(
        masks,
        saved_mask,
    ):
        raise RuntimeError(
            "Canonical evaluation mask differs from saved comparison mask."
        )

    # -------------------------------------------------------------------------
    # Generate V0 using the same canonical test windows.
    # -------------------------------------------------------------------------

    v0_predictions, v0_indices = (
        generate_v0_predictions(
            common_dataset,
            v0_normalization,
        )
    )

    if not np.array_equal(
        v0_indices,
        sample_indices,
    ):
        raise RuntimeError(
            "V0 sample indices do not match canonical test indices."
        )

    # -------------------------------------------------------------------------
    # Apply identical mask to all models.
    # -------------------------------------------------------------------------

    actuals_masked = np.where(
        masks > 0.5,
        actuals,
        np.nan,
    )

    v0_masked = np.where(
        masks > 0.5,
        v0_predictions,
        np.nan,
    )

    v2_masked = np.where(
        masks > 0.5,
        v2_predictions,
        np.nan,
    )

    v2_population_masked = np.where(
        masks > 0.5,
        v2_population_predictions,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Select exactly the same river cells for all models.
    # -------------------------------------------------------------------------

    selected_cells = select_top_river_cells(
        actuals,
        masks,
        river_mask,
    )

    print(
        f"Selected river cells: {len(selected_cells)}"
    )

    # -------------------------------------------------------------------------
    # Evaluate all three models using exactly the same events.
    # -------------------------------------------------------------------------

    v0_results = (
        evaluate_model_events(
            model_name="World Model V0",
            prediction_grid=v0_masked,
            actuals=actuals_masked,
            masks=masks,
            dates=dates,
            selected_cells=selected_cells,
        )
    )

    v2_results = (
        evaluate_model_events(
            model_name="World Model V2",
            prediction_grid=v2_masked,
            actuals=actuals_masked,
            masks=masks,
            dates=dates,
            selected_cells=selected_cells,
        )
    )

    v2_population_results = (
        evaluate_model_events(
            model_name="World Model V2-Population",
            prediction_grid=v2_population_masked,
            actuals=actuals_masked,
            masks=masks,
            dates=dates,
            selected_cells=selected_cells,
        )
    )

    results = {
        "v0": v0_results,
        "v2": v2_results,
        "v2_population": v2_population_results,
    }

    # -------------------------------------------------------------------------
    # Print results.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CANONICAL EVENT COMPARISON")
    print("=" * 80)

    print(
        f"Test start: {dates[0]}"
    )

    print(
        f"Test end: {dates[-1]}"
    )

    print(
        f"Forecast windows: {sample_count}"
    )

    print(
        f"River cells: {len(selected_cells)}"
    )

    print(
        f"Event percentile: {EVENT_PERCENTILE}"
    )

    print(
        f"Minimum event gap: {MIN_EVENT_GAP_DAYS} days"
    )

    print(
        f"Peak match window: -{PEAK_MATCH_WINDOW_BEFORE}/+{PEAK_MATCH_WINDOW_AFTER} days"
    )

    for key in [
        "v0",
        "v2",
        "v2_population",
    ]:
        result = results[
            key
        ]

        print("-" * 80)

        print(
            result["model"]
        )

        print(
            f"Total events: {result['total_events']}"
        )

        print(
            f"Mean relative peak error: {result['mean_relative_peak_error_percent']:.2f}%"
        )

        print(
            f"Median relative peak error: {result['median_relative_peak_error_percent']:.2f}%"
        )

        print(
            f"Mean peak timing error: {result['mean_peak_timing_error_days']:.2f} days"
        )

        print(
            f"Median peak timing error: {result['median_peak_timing_error_days']:.2f} days"
        )

        print(
            f"Severe underprediction count: {result['severe_underprediction_count']}"
        )

        print(
            f"Severe underprediction rate: {result['severe_underprediction_rate_percent']:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Save output.
    # -------------------------------------------------------------------------

    output = {
        "evaluation": (
            "canonical_high_flow_spatial_event_comparison"
        ),
        "test_start_index": int(
            test_start
        ),
        "test_end_index": int(
            test_end
        ),
        "test_start_date": str(
            dates[0]
        ),
        "test_end_date": str(
            dates[-1]
        ),
        "forecast_windows": int(
            sample_count
        ),
        "top_river_cells": int(
            len(selected_cells)
        ),
        "event_percentile": EVENT_PERCENTILE,
        "minimum_event_gap_days": MIN_EVENT_GAP_DAYS,
        "peak_match_window_before_days": PEAK_MATCH_WINDOW_BEFORE,
        "peak_match_window_after_days": PEAK_MATCH_WINDOW_AFTER,
        "severe_underprediction_threshold_ratio": (
            SEVERE_UNDERPREDICTION_RATIO
        ),
        "evaluation_mask": (
            "glofas_discharge_valid_t × river_mask"
        ),
        "v0": v0_results,
        "v2": v2_results,
        "v2_population": v2_population_results,
        "selected_cells": [
            {
                "lat_index": int(
                    row
                ),
                "lon_index": int(
                    col
                ),
                "latitude": float(
                    lat[row]
                ),
                "longitude": float(
                    lon[col]
                ),
            }
            for row, col in selected_cells
        ],
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
            output,
            file,
            indent=2,
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
