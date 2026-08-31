from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def robust_minmax(
    values: np.ndarray,
    valid_mask: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    valid = (
        valid_mask
        & np.isfinite(values)
    )

    result = np.zeros_like(
        values,
        dtype=np.float32,
    )

    if not valid.any():
        return result

    valid_values = values[
        valid
    ].astype(np.float64)

    lower = float(
        np.percentile(
            valid_values,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            valid_values,
            upper_percentile,
        )
    )

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
    ):
        return result

    if upper <= lower:
        result[valid] = 0.0
        return result

    result[valid] = np.clip(
        (
            values[valid]
            - lower
        )
        / (
            upper - lower
        ),
        0.0,
        1.0,
    )

    return result.astype(
        np.float32
    )


def build_population_risk(
    hazard_path: str | Path,
    static_path: str | Path,
    population_weight: float = 1.0,
) -> xr.Dataset:
    hazard_path = Path(
        hazard_path
    )

    static_path = Path(
        static_path
    )

    if not hazard_path.exists():
        raise FileNotFoundError(
            f"Missing hazard dataset: {hazard_path}"
        )

    if not static_path.exists():
        raise FileNotFoundError(
            f"Missing static dataset: {static_path}"
        )

    hazard_ds = xr.open_dataset(
        hazard_path
    )

    hazard = (
        hazard_ds[
            "hydrological_hazard_score"
        ]
        .values
        .astype(np.float32)
    )

    hazard_mask = (
        hazard_ds[
            "discharge_component"
        ]
        .values
    )

    sample = hazard_ds[
        "sample"
    ].values

    forecast_day = hazard_ds[
        "forecast_day"
    ].values

    lat = hazard_ds[
        "lat"
    ].values

    lon = hazard_ds[
        "lon"
    ].values

    hazard_ds.close()

    static_ds = xr.open_zarr(
        static_path,
        consolidated=True,
    )

    population = (
        static_ds[
            "population_density"
        ]
        .values
        .astype(np.float32)
    )

    land_mask = (
        static_ds[
            "land_mask"
        ]
        .values
        .astype(np.float32)
    )

    static_lat = static_ds[
        "lat"
    ].values

    static_lon = static_ds[
        "lon"
    ].values

    static_ds.close()

    if not np.array_equal(
        lat,
        static_lat,
    ):
        raise RuntimeError(
            "Hazard latitude does not match population latitude."
        )

    if not np.array_equal(
        lon,
        static_lon,
    ):
        raise RuntimeError(
            "Hazard longitude does not match population longitude."
        )

    if population.shape != (
        len(lat),
        len(lon),
    ):
        raise RuntimeError(
            f"Unexpected population shape: {population.shape}"
        )

    if not np.isfinite(
        population
    ).all():
        raise RuntimeError(
            "Population density contains NaN or Inf."
        )

    if not np.isfinite(
        hazard_mask[
            np.isfinite(hazard_mask)
        ]
    ).all():
        raise RuntimeError(
            "Hazard validity data contains invalid values."
        )

    population_nonnegative = np.maximum(
        population,
        0.0,
    )

    # Population density is strongly right-skewed, so log1p is used for
    # relative spatial normalization.
    population_log = np.log1p(
        population_nonnegative
    )

    population_valid = (
        (land_mask > 0.5)
        & np.isfinite(
            population_log
        )
    )

    population_component = robust_minmax(
        population_log,
        population_valid,
    )

    population_component = np.broadcast_to(
        population_component[
            None,
            None,
            ...,
        ],
        hazard.shape,
    ).astype(np.float32)

    valid = (
        np.isfinite(
            hazard
        )
        & (
            land_mask[
                None,
                None,
                ...
            ]
            > 0.5
        )
    )

    # Population exposure index:
    #
    #     hazard × normalized population
    #
    # This is an exposure/risk indicator, not an estimate of exact people
    # physically flooded.
    population_exposure_index = (
        hazard
        * population_component
        * float(
            population_weight
        )
    )

    population_exposure_index = np.clip(
        population_exposure_index,
        0.0,
        1.0,
    ).astype(np.float32)

    population_exposure_index = np.where(
        valid,
        population_exposure_index,
        np.nan,
    )

    # Approximate population represented by each 0.1° grid cell.
    #
    # The source is a density surface and the exact interpretation of its units
    # depends on the source product. We therefore keep this as an exposure
    # index rather than claiming exact flooded population counts.
    population_density_output = np.where(
        valid,
        np.broadcast_to(
            population_nonnegative[
                None,
                None,
                ...
            ],
            hazard.shape,
        ),
        np.nan,
    ).astype(np.float32)

    output = xr.Dataset(
        {
            "population_density": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                population_density_output,
            ),
            "population_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                np.where(
                    valid,
                    population_component,
                    np.nan,
                ).astype(np.float32),
            ),
            "hydrological_hazard_score": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                np.where(
                    valid,
                    hazard,
                    np.nan,
                ).astype(np.float32),
            ),
            "population_exposure_index": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                population_exposure_index,
            ),
        },
        coords={
            "sample": sample,
            "forecast_day": forecast_day,
            "lat": lat,
            "lon": lon,
        },
    )

    output[
        "population_density"
    ].attrs["units"] = (
        "source dataset units"
    )

    output[
        "population_component"
    ].attrs["range"] = "0-1"

    output[
        "population_exposure_index"
    ].attrs["range"] = "0-1"

    output[
        "population_exposure_index"
    ].attrs["interpretation"] = (
        "relative hazard-weighted population exposure index"
    )

    output.attrs[
        "model"
    ] = (
        "Bangladesh Flood World Model V2-Population"
    )

    output.attrs[
        "risk_type"
    ] = (
        "population exposure index"
    )

    output.attrs[
        "calibrated_flooded_population"
    ] = "false"

    output.attrs[
        "population_source"
    ] = (
        "data/features/static_v3.zarr/population_density"
    )

    output.attrs[
        "formula"
    ] = (
        "hydrological_hazard_score × normalized_log1p_population_density"
    )

    return output


def summarize_population_risk(
    dataset: xr.Dataset,
) -> dict[str, Any]:
    exposure = (
        dataset[
            "population_exposure_index"
        ]
        .values
    )

    finite = exposure[
        np.isfinite(exposure)
    ]

    if finite.size == 0:
        return {
            "finite_values": 0,
            "mean_exposure_index": None,
            "median_exposure_index": None,
            "max_exposure_index": None,
            "high_exposure_fraction": None,
        }

    return {
        "finite_values": int(
            finite.size
        ),
        "mean_exposure_index": float(
            np.mean(
                finite
            )
        ),
        "median_exposure_index": float(
            np.median(
                finite
            )
        ),
        "max_exposure_index": float(
            np.max(
                finite
            )
        ),
        "high_exposure_fraction": float(
            np.mean(
                finite >= 0.70
            )
        ),
    }


def save_population_risk(
    dataset: xr.Dataset,
    output_path: str | Path,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_netcdf(
        output_path
    )

    summary = summarize_population_risk(
        dataset
    )

    summary[
        "output"
    ] = str(
        output_path
    )

    if metrics_path is not None:
        metrics_path = Path(
            metrics_path
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

    return summary
