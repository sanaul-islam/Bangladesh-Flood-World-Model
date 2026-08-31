from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
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


@pytest.mark.integration
def test_complete_evacuation_pipeline():
    if not DATABASE_PATH.exists():
        pytest.skip(
            "Road database unavailable."
        )

    if not POPULATION_RISK_PATH.exists():
        pytest.skip(
            "Population-risk artifact unavailable."
        )

    from flood_world_model.planning.shelter_ranking import (
        ShelterRanker,
    )

    with ShelterRanker(
        DATABASE_PATH,
        POPULATION_RISK_PATH,
    ) as ranker:

        result = ranker.rank(
            user_latitude=23.8103,
            user_longitude=90.4125,
            forecast_sample=0,
            forecast_day=1,
            candidate_count=5,
        )

    assert (
        "recommended_shelter"
        in result
    )

    assert (
        result["system"]["reachable_shelters"]
        >= 1
    )

    recommendation = result[
        "recommended_shelter"
    ]

    assert (
        recommendation["shelter_id"]
        >= 0
    )

    assert (
        -90.0
        <= recommendation["latitude"]
        <= 90.0
    )

    assert (
        -180.0
        <= recommendation["longitude"]
        <= 180.0
    )

    assert (
        0.0
        <= recommendation["destination_hazard"]
        <= 1.0
    )

    assert (
        0.0
        <= recommendation[
            "destination_population_exposure"
        ]
        <= 1.0
    )

    assert (
        recommendation["road_distance_km"]
        >= 0.0
    )

    assert (
        recommendation["travel_time_min"]
        >= 0.0
    )

    assert (
        recommendation["risk_cost"]
        > 0.0
    )

    assert (
        recommendation["bridge_edges"]
        >= 0
    )

    route = result[
        "route"
    ]

    assert (
        "route"
        in route
    )

    coordinates = route[
        "route"
    ][
        "coordinates"
    ]

    assert len(
        coordinates
    ) >= 2

    assert all(
        len(point) == 2
        for point in coordinates
    )

    assert (
        route["statistics"][
            "road_distance_km"
        ]
        >= 0.0
    )

    assert (
        route["statistics"][
            "risk_cost"
        ]
        > 0.0
    )
