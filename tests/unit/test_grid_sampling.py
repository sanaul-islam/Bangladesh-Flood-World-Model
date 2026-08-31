from __future__ import annotations

import numpy as np

from flood_world_model.planning.grid_sampling import (
    bilinear_sample,
)


def test_bilinear_sampling():
    field = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )

    lat = np.array(
        [0.0, 1.0],
        dtype=np.float64,
    )

    lon = np.array(
        [0.0, 1.0],
        dtype=np.float64,
    )

    value, metadata = bilinear_sample(
        field=field,
        lat_values=lat,
        lon_values=lon,
        latitude=0.5,
        longitude=0.5,
    )

    assert value is not None
    assert abs(value - 1.0) < 1e-6
    assert metadata["method"] == "bilinear"


def test_nearest_valid_fallback():
    field = np.array(
        [
            [np.nan, np.nan],
            [np.nan, 5.0],
        ],
        dtype=np.float32,
    )

    lat = np.array(
        [0.0, 1.0],
        dtype=np.float64,
    )

    lon = np.array(
        [0.0, 1.0],
        dtype=np.float64,
    )

    value, metadata = bilinear_sample(
        field=field,
        lat_values=lat,
        lon_values=lon,
        latitude=0.1,
        longitude=0.1,
    )

    assert value == 5.0
    assert metadata["method"] == "nearest_valid"
