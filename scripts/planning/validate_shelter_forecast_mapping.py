from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "src"
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


def main() -> None:
    print("=" * 80)
    print(
        "SHELTER FORECAST MAPPING VALIDATION"
    )
    print("=" * 80)

    with ShelterRanker(
        DATABASE_PATH,
        POPULATION_RISK_PATH,
    ) as ranker:

        rows = ranker.connection.execute(
            """
            SELECT
                s.shelter_id,
                s.x,
                s.y
            FROM shelters AS s
            ORDER BY s.shelter_id
            """
        ).fetchall()

        valid = 0
        invalid = 0

        for shelter_id, x, y in rows:

            latitude, longitude = (
                ranker.router.xy_to_latlon(
                    float(x),
                    float(y),
                )
            )

            try:
                values = (
                    ranker.destination_values(
                        latitude=latitude,
                        longitude=longitude,
                        forecast_sample=0,
                        forecast_day=1,
                    )
                )

            except RuntimeError as error:
                invalid += 1

                print(
                    f"INVALID shelter={shelter_id}: "
                    f"{error}"
                )

                continue

            valid += 1

            print(
                f"shelter={shelter_id} "
                f"hazard={values['hazard_score']:.4f} "
                f"exposure={values['population_exposure']:.4f} "
                f"method={values['hazard_sampling']['method']}"
            )

    print("=" * 80)
    print(
        "SHELTER FORECAST VALIDATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Valid: {valid}"
    )

    print(
        f"Invalid: {invalid}"
    )

    if valid == 0:
        raise RuntimeError(
            "No shelters have valid forecast mapping."
        )


if __name__ == "__main__":
    main()
