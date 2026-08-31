from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset
from flood_world_model.models.world_model import FloodWorldModel
from flood_world_model.utils.paths import NORMALIZATION_PATH, OUTPUT_DIR, V0_CHECKPOINT

DEVICE = torch.device("cpu")
CHECKPOINT_PATH = V0_CHECKPOINT
OUTPUT_PATH = OUTPUT_DIR / "flood_event_analysis.json"


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = FloodWorldModel(dynamic_channels=checkpoint["dynamic_channels"], static_channels=checkpoint["static_channels"], hidden_channels=checkpoint["hidden_channels"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def group_events(dates, values, threshold):
    above = values >= threshold
    events = []
    start = None

    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            events.append((start, i - 1))
            start = None

    if start is not None:
        events.append((start, len(values) - 1))

    return events


def main():
    print("=" * 80)
    print("FLOOD EVENT ANALYSIS")
    print("=" * 80)

    model = load_model()

    dataset = FloodWorldModelDataset(split="test", history_days=14, forecast_days=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
        norm = json.load(f)

    discharge_mean = float(norm["river_discharge"]["mean"])
    discharge_std = max(float(norm["river_discharge"]["std"]), 1e-8)

    test_indices = np.load("data/features/training_v3/test_indices.npy").astype(np.int64)
    dynamic_time = np.array(pd.date_range("2015-01-01", "2026-06-01", freq="D"))

    daily_actual = []
    daily_prediction = []
    daily_persistence = []
    daily_dates = []

    with torch.no_grad():
        for i, (dynamic, static, target, target_mask) in enumerate(loader):
            dynamic = dynamic.to(DEVICE)
            static = static.to(DEVICE)
            target = target[:, 0].to(DEVICE)
            target_mask = target_mask[:, 0].to(DEVICE)

            prediction = model(dynamic, static)
            persistence = dynamic[:, -1, 5:6]

            river_mask = static[:, 3:4]
            mask = target_mask * river_mask

            p = prediction.squeeze().cpu().numpy()
            t = target.squeeze().cpu().numpy()
            q = persistence.squeeze().cpu().numpy()
            m = mask.squeeze().cpu().numpy() > 0.5

            if m.any():
                p = p[m] * discharge_std + discharge_mean
                t = t[m] * discharge_std + discharge_mean
                q = q[m] * discharge_std + discharge_mean

                daily_prediction.append(float(np.max(p)))
                daily_actual.append(float(np.max(t)))
                daily_persistence.append(float(np.max(q)))
            else:
                daily_prediction.append(np.nan)
                daily_actual.append(np.nan)
                daily_persistence.append(np.nan)

            target_index = int(test_indices[i]) + 14
            daily_dates.append(dynamic_time[target_index])

            if i == 1 or (i + 1) % 200 == 0:
                print(f"Processed {i + 1}/{len(loader)} test windows")

    df = pd.DataFrame({
        "date": pd.to_datetime(daily_dates),
        "actual_peak": daily_actual,
        "predicted_peak": daily_prediction,
        "persistence_peak": daily_persistence,
    })

    df = df.dropna().sort_values("date").reset_index(drop=True)

    threshold_95 = float(df["actual_peak"].quantile(0.95))
    threshold_99 = float(df["actual_peak"].quantile(0.99))

    print(f"Daily maximum discharge 95th percentile: {threshold_95:.4f} m3/s")
    print(f"Daily maximum discharge 99th percentile: {threshold_99:.4f} m3/s")

    events = group_events(
        df["date"].values,
        df["actual_peak"].values,
        threshold_95,
    )

    event_results = []

    for event_number, (start, end) in enumerate(events, start=1):
        event = df.iloc[start:end + 1]

        actual_peak_index = event["actual_peak"].idxmax()
        predicted_peak_index = event["predicted_peak"].idxmax()
        persistence_peak_index = event["persistence_peak"].idxmax()

        actual_peak = float(event["actual_peak"].max())
        predicted_peak = float(event["predicted_peak"].max())
        persistence_peak = float(event["persistence_peak"].max())

        actual_date = pd.Timestamp(df.loc[actual_peak_index, "date"])
        predicted_date = pd.Timestamp(df.loc[predicted_peak_index, "date"])
        persistence_date = pd.Timestamp(df.loc[persistence_peak_index, "date"])

        event_results.append({
            "event": event_number,
            "start": str(event["date"].min().date()),
            "end": str(event["date"].max().date()),
            "duration_days": int(len(event)),
            "actual_peak_m3_s": actual_peak,
            "predicted_peak_m3_s": predicted_peak,
            "persistence_peak_m3_s": persistence_peak,
            "peak_error_m3_s": predicted_peak - actual_peak,
            "peak_absolute_error_m3_s": abs(predicted_peak - actual_peak),
            "peak_relative_error": abs(predicted_peak - actual_peak) / max(abs(actual_peak), 1e-8),
            "peak_timing_error_days": int((predicted_date - actual_date).days),
            "persistence_timing_error_days": int((persistence_date - actual_date).days),
        })

    if event_results:
        mean_peak_error = float(np.mean([x["peak_absolute_error_m3_s"] for x in event_results]))
        mean_timing_error = float(np.mean([abs(x["peak_timing_error_days"]) for x in event_results]))
        severe_underprediction_rate = float(np.mean([x["predicted_peak_m3_s"] < 0.9 * x["actual_peak_m3_s"] for x in event_results]))
    else:
        mean_peak_error = float("nan")
        mean_timing_error = float("nan")
        severe_underprediction_rate = float("nan")

    summary = {
        "test_days": int(len(df)),
        "threshold_95_m3_s": threshold_95,
        "threshold_99_m3_s": threshold_99,
        "number_of_95th_percentile_events": len(event_results),
        "mean_event_peak_absolute_error_m3_s": mean_peak_error,
        "mean_absolute_peak_timing_error_days": mean_timing_error,
        "severe_underprediction_rate": severe_underprediction_rate,
        "events": event_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("EVENT SUMMARY")
    print("=" * 80)
    print(f"Events: {len(event_results)}")
    print(f"Mean peak absolute error: {mean_peak_error:.4f} m3/s")
    print(f"Mean absolute timing error: {mean_timing_error:.4f} days")
    print(f"Severe underprediction rate: {severe_underprediction_rate:.2%}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()