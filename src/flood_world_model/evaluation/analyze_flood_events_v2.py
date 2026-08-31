from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

INPUT_PATH = Path("outputs/flood_event_analysis.json")
OUTPUT_PATH = Path("outputs/flood_event_analysis_v2.json")


def main():
    print("=" * 80)
    print("FLOOD EVENT ANALYSIS V2")
    print("=" * 80)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    events = report.get("events", [])

    if not events:
        print("No events found.")
        return

    rows = []

    for event in events:
        rows.append({
            "event": event["event"],
            "start": event["start"],
            "end": event["end"],
            "duration_days": event["duration_days"],
            "actual_peak_m3_s": event["actual_peak_m3_s"],
            "predicted_peak_m3_s": event["predicted_peak_m3_s"],
            "persistence_peak_m3_s": event["persistence_peak_m3_s"],
            "peak_absolute_error_m3_s": event["peak_absolute_error_m3_s"],
            "peak_relative_error": event["peak_relative_error"],
            "peak_timing_error_days": event["peak_timing_error_days"],
            "persistence_timing_error_days": event["persistence_timing_error_days"],
        })

    df = pd.DataFrame(rows)

    print("\nEvent details:")
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("EVENT STATISTICS")
    print("=" * 80)

    print(f"Number of events: {len(df)}")
    print(f"Mean peak error: {df['peak_absolute_error_m3_s'].mean():.2f} m3/s")
    print(f"Median peak error: {df['peak_absolute_error_m3_s'].median():.2f} m3/s")
    print(f"Mean relative peak error: {df['peak_relative_error'].mean():.2%}")
    print(f"Median relative peak error: {df['peak_relative_error'].median():.2%}")
    print(f"Mean absolute timing error: {df['peak_timing_error_days'].abs().mean():.2f} days")
    print(f"Maximum absolute timing error: {df['peak_timing_error_days'].abs().max():.2f} days")

    severe_underprediction = (
        df["predicted_peak_m3_s"]
        < 0.9 * df["actual_peak_m3_s"]
    )

    print(f"Severe underprediction events: {int(severe_underprediction.sum())}/{len(df)}")

    df["peak_ratio"] = (
        df["predicted_peak_m3_s"]
        / df["actual_peak_m3_s"].replace(0, np.nan)
    )

    print(f"Mean predicted/actual peak ratio: {df['peak_ratio'].mean():.4f}")

    output = {
        "number_of_events": int(len(df)),
        "mean_peak_error_m3_s": float(df["peak_absolute_error_m3_s"].mean()),
        "median_peak_error_m3_s": float(df["peak_absolute_error_m3_s"].median()),
        "mean_relative_peak_error": float(df["peak_relative_error"].mean()),
        "median_relative_peak_error": float(df["peak_relative_error"].median()),
        "mean_absolute_timing_error_days": float(df["peak_timing_error_days"].abs().mean()),
        "max_absolute_timing_error_days": float(df["peak_timing_error_days"].abs().max()),
        "severe_underprediction_events": int(severe_underprediction.sum()),
        "events": rows,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()