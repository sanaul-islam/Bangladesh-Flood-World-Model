from __future__ import annotations

import os
from pathlib import Path

import xarray as xr

from flood_world_model.inference.rollout import V0Rollout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def main():
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("V0 SEVEN-DAY ROLLOUT")
    print("=" * 80)

    dynamic = xr.open_zarr(
        PROJECT_ROOT / "data/features/dynamic_core_v2.zarr",
        consolidated=True,
    )

    end_index = dynamic.sizes["time"]

    last_date = dynamic.time.values[
        end_index - 1
    ]

    print(f"Forecast start state: {last_date}")

    dynamic.close()

    rollout = V0Rollout(
        PROJECT_ROOT
    )

    result = rollout.rollout(
        end_index=end_index,
        days=7,
    )

    output_path = (
        PROJECT_ROOT
        / "outputs/predictions/v0_7day_forecast.nc"
    )

    rollout.save(
        result,
        output_path,
    )

    print("✅ Seven-day rollout complete.")


if __name__ == "__main__":
    main()