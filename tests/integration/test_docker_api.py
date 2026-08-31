from __future__ import annotations

import os

import pytest
import requests


BASE_URL = os.getenv(
    "FLOOD_API_BASE_URL",
    "http://127.0.0.1:8000",
)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv(
        "RUN_DOCKER_TESTS"
    )
    != "1",
    reason="Docker service test disabled.",
)
def test_docker_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "status"
    ] == "ok"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv(
        "RUN_DOCKER_TESTS"
    )
    != "1",
    reason="Docker service test disabled.",
)
def test_docker_forecast():
    response = requests.get(
        f"{BASE_URL}/api/v1/forecast",
        timeout=10,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "latitude_points"
    ] == 60

    assert payload[
        "longitude_points"
    ] == 45

    assert len(
        payload[
            "forecast_days"
        ]
    ) == 7


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv(
        "RUN_DOCKER_TESTS"
    )
    != "1",
    reason="Docker service test disabled.",
)
def test_docker_evacuate():
    response = requests.post(
        f"{BASE_URL}/api/v1/evacuate",
        json={
            "latitude": 23.8103,
            "longitude": 90.4125,
            "forecast_sample": 0,
            "forecast_day": 1,
            "candidate_shelters": 5,
        },
        timeout=120,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "status"
    ] == "ok"

    result = payload[
        "result"
    ]

    assert (
        result[
            "system"
        ][
            "reachable_shelters"
        ]
        >= 1
    )

    assert (
        result[
            "recommended_shelter"
        ][
            "shelter_id"
        ]
        >= 0
    )
