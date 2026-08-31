from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT / "src"
    ),
)

from flood_world_model.planning.shelter_ranking import (
    ShelterRanker,
)


DATABASE_PATH = (
    PROJECT_ROOT
    / "data/processed/road_network.sqlite"
)

POPULATION_RISK_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_population_population_risk.nc"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/routes/"
    "shelter_recommendation.json"
)

USER_LATITUDE = 23.8103
USER_LONGITUDE = 90.4125

FORECAST_SAMPLE = 0
FORECAST_DAY = 1

CANDIDATE_COUNT = 24


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL")
    print("RISK-AWARE AUTOMATIC EVACUATION RECOMMENDATION")
    print("=" * 80)

    print(
        f"User latitude: {USER_LATITUDE}"
    )

    print(
        f"User longitude: {USER_LONGITUDE}"
    )

    print(
        f"Forecast sample: {FORECAST_SAMPLE}"
    )

    print(
        f"Forecast day: {FORECAST_DAY}"
    )

    print(
        f"Candidate shelters: {CANDIDATE_COUNT}"
    )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing database: {DATABASE_PATH}"
        )

    if not POPULATION_RISK_PATH.exists():
        raise FileNotFoundError(
            f"Missing population-risk dataset: {POPULATION_RISK_PATH}"
        )

    with ShelterRanker(
        database_path=DATABASE_PATH,
        population_risk_path=POPULATION_RISK_PATH,
    ) as ranker:

        result = ranker.rank(
            user_latitude=USER_LATITUDE,
            user_longitude=USER_LONGITUDE,
            forecast_sample=FORECAST_SAMPLE,
            forecast_day=FORECAST_DAY,
            candidate_count=CANDIDATE_COUNT,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    best = result[
        "recommended_shelter"
    ]

    print("=" * 80)
    print("EVACUATION RECOMMENDATION COMPLETE")
    print("=" * 80)

    print(
        f"Recommended shelter: {best['shelter_id']}"
    )

    print(
        f"Latitude: {best['latitude']:.6f}"
    )

    print(
        f"Longitude: {best['longitude']:.6f}"
    )

    print(
        f"Road distance: {best['road_distance_km']:.2f} km"
    )

    print(
        f"Travel time: {best['travel_time_min']:.2f} min"
    )

    print(
        f"Risk cost: {best['risk_cost']:.2f}"
    )

    print(
        f"Destination hazard: {best['destination_hazard']:.4f}"
    )

    print(
        f"Destination population component: "
        f"{best['destination_population_component']:.4f}"
    )

    print(
        f"Destination population exposure: "
        f"{best['destination_population_exposure']:.4f}"
    )

    print(
        f"Population density: "
        f"{best['destination_population_density']:.4f}"
    )

    print(
        f"Maximum route flood risk: "
        f"{best['maximum_flood_risk']:.4f}"
    )

    print(
        f"Maximum route uncertainty: "
        f"{best['maximum_uncertainty']:.4f}"
    )

    print(
        f"Maximum bridge risk: "
        f"{best['maximum_bridge_risk']:.4f}"
    )

    print(
        f"Bridge edges: "
        f"{best['bridge_edges']:,}"
    )

    print(
        f"Combined score: "
        f"{best['combined_score']:.4f}"
    )

    print(
        f"Availability: "
        f"{best['availability']}"
    )

    print(
        f"Capacity: "
        f"{best['capacity']}"
    )

    print(
        f"Reachable shelters evaluated: "
        f"{result['system']['reachable_shelters']}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
