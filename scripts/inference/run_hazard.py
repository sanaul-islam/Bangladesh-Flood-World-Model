from __future__ import annotations

import os
from pathlib import Path

from flood_world_model.hazard.hazard_model import HazardModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def main():
    print("=" * 80)
    print("HAZARD GENERATION")
    print("=" * 80)

    forecast = PROJECT_ROOT / "outputs/predictions/v0_7day_forecast.nc"
    output = PROJECT_ROOT / "outputs/hazard/v0_7day_hazard.nc"

    model = HazardModel(
        PROJECT_ROOT
    )

    model.compute(
        forecast,
        output,
    )

    print("✅ Hazard generation complete.")


if __name__ == "__main__":
    main()