from flood_world_model.api.config import (
    MAX_CANDIDATE_SHELTERS,
    MAX_CONCURRENT_PLANNING_REQUESTS,
    MAX_ROUTE_DISTANCE_KM,
    MAX_ROUTE_EXPANDED_NODES,
)


def test_production_limits_are_valid():
    assert MAX_CANDIDATE_SHELTERS >= 1
    assert MAX_CANDIDATE_SHELTERS <= 24

    assert MAX_ROUTE_DISTANCE_KM > 0.0
    assert MAX_ROUTE_EXPANDED_NODES > 100

    assert (
        MAX_CONCURRENT_PLANNING_REQUESTS
        >= 1
    )
