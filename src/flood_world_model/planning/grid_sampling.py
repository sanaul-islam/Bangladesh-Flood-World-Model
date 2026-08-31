from __future__ import annotations

import math
from typing import Any

import numpy as np


def _fractional_index(
    values: np.ndarray,
    coordinate: float,
) -> tuple[int, int, float]:
    if coordinate <= values[0]:
        return (
            0,
            0,
            0.0,
        )

    if coordinate >= values[-1]:
        last = len(values) - 1

        return (
            last,
            last,
            0.0,
        )

    upper = int(
        np.searchsorted(
            values,
            coordinate,
            side="right",
        )
    )

    lower = upper - 1

    width = (
        values[upper]
        - values[lower]
    )

    if width <= 0:
        return (
            lower,
            lower,
            0.0,
        )

    fraction = (
        coordinate
        - values[lower]
    ) / width

    return (
        lower,
        upper,
        float(
            fraction
        ),
    )


def bilinear_sample(
    field: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    latitude: float,
    longitude: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[
    float | None,
    dict[str, Any],
]:
    lat_values = np.asarray(
        lat_values,
        dtype=np.float64,
    )

    lon_values = np.asarray(
        lon_values,
        dtype=np.float64,
    )

    i0, i1, fy = _fractional_index(
        lat_values,
        latitude,
    )

    j0, j1, fx = _fractional_index(
        lon_values,
        longitude,
    )

    corners = [
        (
            i0,
            j0,
            (1.0 - fy)
            * (1.0 - fx),
        ),
        (
            i0,
            j1,
            (1.0 - fy)
            * fx,
        ),
        (
            i1,
            j0,
            fy
            * (1.0 - fx),
        ),
        (
            i1,
            j1,
            fy
            * fx,
        ),
    ]

    weighted_sum = 0.0
    weight_sum = 0.0

    all_valid = True

    for i, j, weight in corners:

        value = float(
            field[
                i,
                j,
            ]
        )

        is_valid = math.isfinite(
            value
        )

        if valid_mask is not None:
            is_valid = (
                is_valid
                and bool(
                    valid_mask[
                        i,
                        j
                    ]
                )
            )

        if not is_valid:
            all_valid = False
            break

        weighted_sum += (
            value
            * weight
        )

        weight_sum += weight

    if all_valid and weight_sum > 0.0:
        return (
            float(
                weighted_sum
                / weight_sum
            ),
            {
                "method": "bilinear",
                "lat_index_0": i0,
                "lat_index_1": i1,
                "lon_index_0": j0,
                "lon_index_1": j1,
            },
        )

    # Nearest valid-cell fallback.
    finite = np.isfinite(
        field
    )

    if valid_mask is not None:
        finite &= valid_mask

    rows, cols = np.where(
        finite
    )

    if len(rows) == 0:
        return (
            None,
            {
                "method": "unavailable"
            },
        )

    mean_lat = float(
        np.mean(
            lat_values
        )
    )

    cos_lat = math.cos(
        math.radians(
            mean_lat
        )
    )

    lat_delta = (
        lat_values[
            rows
        ]
        - latitude
    )

    lon_delta = (
        (
            lon_values[
                cols
            ]
            - longitude
        )
        * cos_lat
    )

    distances = (
        lat_delta**2
        + lon_delta**2
    )

    nearest = int(
        np.argmin(
            distances
        )
    )

    row = int(
        rows[
            nearest
        ]
    )

    col = int(
        cols[
            nearest
        ]
    )

    return (
        float(
            field[
                row,
                col,
            ]
        ),
        {
            "method": "nearest_valid",
            "lat_index": row,
            "lon_index": col,
            "distance_degrees": float(
                math.sqrt(
                    distances[
                        nearest
                    ]
                )
            ),
        },
    )
