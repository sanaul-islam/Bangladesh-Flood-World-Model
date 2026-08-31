from __future__ import annotations

from fastapi.testclient import TestClient

from flood_world_model.api.app import app


def test_health():
    with TestClient(
        app
    ) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload[
            "status"
        ] == "ok"

        assert payload[
            "service"
        ] == "bangladesh-flood-world-model"


def test_forecast_endpoint():
    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/forecast"
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


def test_hazard_endpoint():
    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/hazard",
            params={
                "latitude": 23.8103,
                "longitude": 90.4125,
                "forecast_sample": 0,
                "forecast_day": 1,
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert (
            0.0
            <= payload[
                "hazard_score"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= payload[
                "population_exposure"
            ]
            <= 1.0
        )


def test_shelter_endpoint():
    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/shelters"
        )

        assert response.status_code == 200

        payload = response.json()

        assert (
            payload[
                "total_shelters"
            ] == 281
        )

        assert (
            payload[
                "mapped_shelters"
            ] == 274
        )


def test_route_endpoint():
    with TestClient(
        app
    ) as client:

        response = client.post(
            "/api/v1/route",
            json={
                "start_latitude": 23.8103,
                "start_longitude": 90.4125,
                "goal_latitude": 23.80208345,
                "goal_longitude": 90.40965735,
                "max_distance_km": 30.0,
                "max_expanded_nodes": 100000,
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload[
            "status"
        ] == "ok"

        route = payload[
            "result"
        ]

        assert len(
            route[
                "route"
            ][
                "coordinates"
            ]
        ) >= 2

        assert (
            route[
                "statistics"
            ][
                "road_distance_km"
            ] > 0.0
        )


def test_evacuate_endpoint():
    with TestClient(app) as client:

        response = client.post(
            "/api/v1/evacuate",
            json={
                "latitude": 23.8103,
                "longitude": 90.4125,
                "forecast_sample": 0,
                "forecast_day": 1,
                "candidate_shelters": 5,
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["status"] == "ok"

        result = payload["result"]

        assert (
            result["system"]["reachable_shelters"]
            >= 1
        )

        recommendation = (
            result["recommended_shelter"]
        )

        assert (
            recommendation["shelter_id"]
            >= 0
        )

        assert (
            recommendation["road_distance_km"]
            > 0
        )

        assert (
            0.0
            <= recommendation[
                "destination_hazard"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= recommendation[
                "destination_population_exposure"
            ]
            <= 1.0
        )

        assert len(
            result["route"]["route"]["coordinates"]
        ) >= 2

def test_invalid_location():
    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/hazard",
            params={
                "latitude": 300.0,
                "longitude": 90.0,
            },
        )

        assert response.status_code == 422


def test_lifecycle_closes_service():
    with TestClient(
        app
    ) as client:

        service = getattr(
            app.state,
            "flood_service",
            None,
        )

        assert service is not None
        assert service.ready is True

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

    service = getattr(
        app.state,
        "flood_service",
        None,
    )

    assert service is None
