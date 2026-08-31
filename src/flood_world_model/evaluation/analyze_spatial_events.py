from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset
from flood_world_model.models.world_model import FloodWorldModel
from flood_world_model.utils.paths import DYNAMIC_CORE, NORMALIZATION_PATH, OUTPUT_DIR, STATIC_CORE, TEST_INDEX_PATH, V0_CHECKPOINT

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = V0_CHECKPOINT
DYNAMIC_PATH = DYNAMIC_CORE
STATIC_PATH = STATIC_CORE
OUTPUT_PATH = OUTPUT_DIR / "spatial_event_analysis_v2.json"


def find_events(values, dates, percentile=95.0, min_duration=2, gap_tolerance=2):
    valid = np.isfinite(values)

    if valid.sum() == 0:
        return []

    threshold = np.nanpercentile(values[valid], percentile)
    above = valid & (values >= threshold)

    raw_events = []
    start = None
    last_above = None

    for i in range(len(values)):
        if above[i]:
            if start is None:
                start = i
            last_above = i
        elif start is not None and i - last_above > gap_tolerance:
            end = last_above
            if end - start + 1 >= min_duration:
                raw_events.append((start, end))
            start = None
            last_above = None

    if start is not None:
        end = last_above
        if end - start + 1 >= min_duration:
            raw_events.append((start, end))

    return [(dates[s], dates[e], s, e, threshold) for s, e in raw_events]


def calculate_event_metrics(actual, predicted, dates):
    events = find_events(actual, dates)

    results = []

    for event_number, (start_date, end_date, start_idx, end_idx, threshold) in enumerate(events, start=1):
        actual_event = actual[start_idx:end_idx + 1]
        predicted_event = predicted[start_idx:end_idx + 1]
        event_dates = dates[start_idx:end_idx + 1]

        actual_peak_idx = int(np.nanargmax(actual_event))
        actual_peak = float(actual_event[actual_peak_idx])
        actual_peak_date = event_dates[actual_peak_idx]

        predicted_peak_idx = int(np.nanargmax(predicted_event))
        predicted_peak = float(predicted_event[predicted_peak_idx])
        predicted_peak_date = event_dates[predicted_peak_idx]

        peak_error = predicted_peak - actual_peak
        relative_error = abs(peak_error) / max(abs(actual_peak), 1e-8)

        timing_error = float((predicted_peak_date - actual_peak_date) / np.timedelta64(1, "D"))

        actual_mean = float(np.nanmean(actual_event))
        predicted_mean = float(np.nanmean(predicted_event))

        results.append({
            "event": event_number,
            "start": str(pd.Timestamp(start_date).date()),
            "end": str(pd.Timestamp(end_date).date()),
            "duration_days": int(end_idx - start_idx + 1),
            "threshold_m3_s": float(threshold),
            "actual_peak_m3_s": actual_peak,
            "predicted_peak_m3_s": predicted_peak,
            "peak_error_m3_s": float(peak_error),
            "peak_absolute_error_m3_s": float(abs(peak_error)),
            "peak_relative_error": float(relative_error),
            "actual_peak_date": str(pd.Timestamp(actual_peak_date).date()),
            "predicted_peak_date": str(pd.Timestamp(predicted_peak_date).date()),
            "peak_timing_error_days": timing_error,
            "actual_event_mean_m3_s": actual_mean,
            "predicted_event_mean_m3_s": predicted_mean
        })

    return results


def main():
    print("=" * 80)
    print("SPATIAL FLOOD EVENT ANALYSIS V2")
    print("=" * 80)

    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
        normalization = json.load(f)

    discharge_mean = float(normalization["river_discharge"]["mean"])
    discharge_std = max(float(normalization["river_discharge"]["std"]), 1e-8)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model = FloodWorldModel(
        dynamic_channels=checkpoint["dynamic_channels"],
        static_channels=checkpoint["static_channels"],
        hidden_channels=checkpoint["hidden_channels"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dynamic = xr.open_zarr(DYNAMIC_PATH, consolidated=True)
    static = xr.open_zarr(STATIC_PATH, consolidated=True)

    lat = dynamic.lat.values
    lon = dynamic.lon.values

    river_mask = static["river_mask"].values > 0.5

    dataset = FloodWorldModelDataset(split="test", history_days=14, forecast_days=1)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_indices = np.load(TEST_INDEX_PATH).astype(np.int64)

    actual_maps = []
    predicted_maps = []
    dates = []

    print(f"Test samples: {len(dataset):,}")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            dynamic_x, static_x, target, target_mask = batch

            dynamic_x = dynamic_x.to(DEVICE)
            static_x = static_x.to(DEVICE)
            target = target[:, 0].to(DEVICE)
            target_mask = target_mask[:, 0].to(DEVICE)

            prediction = model(dynamic_x, static_x)

            actual = target[0, 0].cpu().numpy()
            predicted = prediction[0, 0].cpu().numpy()
            mask = target_mask[0, 0].cpu().numpy() > 0.5

            actual = actual * discharge_std + discharge_mean
            predicted = predicted * discharge_std + discharge_mean

            valid = (
                mask
                & river_mask
                & np.isfinite(actual)
                & np.isfinite(predicted)
            )

            actual_map = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
            predicted_map = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)

            actual_map[valid] = actual[valid]
            predicted_map[valid] = predicted[valid]

            actual_maps.append(actual_map)
            predicted_maps.append(predicted_map)

            date_index = test_indices[i] + 14
            dates.append(pd.Timestamp(dynamic.time.values[date_index]))

            if i == 0 or (i + 1) % 200 == 0:
                print(f"Processed {i + 1}/{len(loader)}")

    actual_array = np.stack(actual_maps)
    predicted_array = np.stack(predicted_maps)
    dates = np.array(dates, dtype="datetime64[ns]")

    # --------------------------------------------------------
    # Find important river cells using actual peak discharge.
    # --------------------------------------------------------

    cell_max = np.nanmax(actual_array, axis=0)

    candidates = np.argwhere(np.isfinite(cell_max))

    ranked_cells = sorted(
        candidates,
        key=lambda x: cell_max[x[0], x[1]],
        reverse=True
    )

    top_cells = ranked_cells[:25]

    all_cell_results = []

    print("\nTop 25 river cells:")

    for rank, (lat_i, lon_i) in enumerate(top_cells, start=1):
        actual_ts = actual_array[:, lat_i, lon_i]
        predicted_ts = predicted_array[:, lat_i, lon_i]

        valid = np.isfinite(actual_ts) & np.isfinite(predicted_ts)

        if valid.sum() < 30:
            continue

        actual_valid = actual_ts[valid]
        predicted_valid = predicted_ts[valid]
        dates_valid = dates[valid]

        event_results = calculate_event_metrics(
            actual_valid,
            predicted_valid,
            dates_valid
        )

        cell_summary = {
            "rank": rank,
            "lat": float(lat[lat_i]),
            "lon": float(lon[lon_i]),
            "test_period_max_actual_m3_s": float(np.nanmax(actual_valid)),
            "number_of_events": len(event_results),
            "events": event_results
        }

        if event_results:
            cell_summary["mean_event_relative_peak_error"] = float(
                np.mean([e["peak_relative_error"] for e in event_results])
            )

            cell_summary["mean_absolute_timing_error_days"] = float(
                np.mean([abs(e["peak_timing_error_days"]) for e in event_results])
            )

            cell_summary["severe_underprediction_rate"] = float(
                np.mean([
                    e["predicted_peak_m3_s"] < 0.90 * e["actual_peak_m3_s"]
                    for e in event_results
                ])
            )

        all_cell_results.append(cell_summary)

        print(
            f"{rank:02d} lat={lat[lat_i]:.3f} lon={lon[lon_i]:.3f} "
            f"events={len(event_results)} "
            f"max_actual={cell_summary['test_period_max_actual_m3_s']:.1f}"
        )

    cell_event_errors = []
    cell_timing_errors = []
    severe_rates = []

    for cell in all_cell_results:
        for event in cell["events"]:
            cell_event_errors.append(event["peak_relative_error"])
            cell_timing_errors.append(abs(event["peak_timing_error_days"]))

        if "severe_underprediction_rate" in cell:
            severe_rates.append(cell["severe_underprediction_rate"])

    summary = {
        "cells_analyzed": len(all_cell_results),
        "total_events": int(sum(len(c["events"]) for c in all_cell_results)),
        "mean_relative_peak_error": float(np.mean(cell_event_errors)) if cell_event_errors else None,
        "median_relative_peak_error": float(np.median(cell_event_errors)) if cell_event_errors else None,
        "mean_absolute_timing_error_days": float(np.mean(cell_timing_errors)) if cell_timing_errors else None,
        "median_absolute_timing_error_days": float(np.median(cell_timing_errors)) if cell_timing_errors else None,
        "severe_underprediction_rate": float(np.mean(severe_rates)) if severe_rates else None,
        "cells": all_cell_results
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SPATIAL EVENT SUMMARY V2")
    print("=" * 80)
    print(f"Cells analyzed: {summary['cells_analyzed']}")
    print(f"Total events: {summary['total_events']}")
    print(f"Mean relative peak error: {summary['mean_relative_peak_error']:.2%}")
    print(f"Median relative peak error: {summary['median_relative_peak_error']:.2%}")
    print(f"Mean absolute timing error: {summary['mean_absolute_timing_error_days']:.2f} days")
    print(f"Median absolute timing error: {summary['median_absolute_timing_error_days']:.2f} days")
    print(f"Severe underprediction rate: {summary['severe_underprediction_rate']:.2%}")
    print(f"Saved: {OUTPUT_PATH}")

    dynamic.close()
    static.close()


if __name__ == "__main__":
    main()