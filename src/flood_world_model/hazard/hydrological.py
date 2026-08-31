
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

    valid_mask = (
        valid_mask
        & np.isfinite(values)
    )

    result = np.zeros_like(
        values,
        dtype=np.float32,
    )

    if not valid_mask.any():
        return result

    valid_values = values[
        valid_mask
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

    if not np.isfinite(
        lower
    ) or not np.isfinite(
        upper
    ):
        return result

    if upper <= lower:
        result[valid_mask] = 0.0
        return result

    result[valid_mask] = np.clip(
        (
            values[valid_mask]
            - lower
        )
        / (
            upper - lower
        ),
        0.0,
        1.0,
    )

    return result


def inverse_robust_minmax(
    values: np.ndarray,
) -> np.ndarray:
    return (
        1.0 - values
    ).astype(np.float32)


def build_hydrological_hazard(
    uncertainty_path: str | Path,
    dynamic_path: str | Path,
    static_path: str | Path,
    sample_indices: np.ndarray,
    weights: dict[str, float] | None = None,
) -> xr.Dataset:
    uncertainty_path = Path(
        uncertainty_path
    )

    dynamic_path = Path(
        dynamic_path
    )

    static_path = Path(
        static_path
    )

    if weights is None:
        weights = {
            "discharge": 0.40,
            "uncertainty": 0.20,
            "rainfall": 0.15,
            "elevation": 0.10,
            "river_distance": 0.10,
            "slope": 0.05,
        }

    required_weight_names = {
        "discharge",
        "uncertainty",
        "rainfall",
        "elevation",
        "river_distance",
        "slope",
    }

    if set(weights) != required_weight_names:
        raise ValueError(
            "Weights must contain exactly: "
            + ", ".join(
                sorted(
                    required_weight_names
                )
            )
        )

    weight_sum = float(
        sum(
            weights.values()
        )
    )

    if weight_sum <= 0.0:
        raise ValueError(
            "Hazard weights must sum to a positive value."
        )

    weights = {
        key: float(
            value / weight_sum
        )
        for key, value in weights.items()
    }

    uncertainty_ds = xr.open_dataset(
        uncertainty_path
    )

    p50 = (
        uncertainty_ds[
            "predicted_discharge_p50"
        ]
        .values
        .astype(np.float32)
    )

    p90 = (
        uncertainty_ds[
            "predicted_discharge_p90"
        ]
        .values
        .astype(np.float32)
    )

    forecast_mask = (
        uncertainty_ds[
            "evaluation_mask"
        ]
        .values
        .astype(np.float32)
    )

    sample = uncertainty_ds[
        "sample"
    ].values

    forecast_day = uncertainty_ds[
        "forecast_day"
    ].values

    lat = uncertainty_ds[
        "lat"
    ].values

    lon = uncertainty_ds[
        "lon"
    ].values

    uncertainty_ds.close()

    if p50.ndim != 4:
        raise RuntimeError(
            "Expected P50 shape [sample, forecast_day, lat, lon]."
        )

    if p90.shape != p50.shape:
        raise RuntimeError(
            "P90 and P50 shapes differ."
        )

    if forecast_mask.shape != p50.shape:
        raise RuntimeError(
            "Forecast mask shape differs from forecast shape."
        )

    sample_indices = np.asarray(
        sample_indices,
        dtype=np.int64,
    )

    if len(sample_indices) != p50.shape[0]:
        raise RuntimeError(
            "sample_indices length does not match forecast samples."
        )

    dynamic_ds = xr.open_zarr(
        dynamic_path,
        consolidated=True,
    )

    precipitation_all = (
        dynamic_ds[
            "precipitation"
        ]
        .values
        .astype(np.float32)
    )

    dynamic_lat = dynamic_ds[
        "lat"
    ].values

    dynamic_lon = dynamic_ds[
        "lon"
    ].values

    total_time = dynamic_ds.sizes[
        "time"
    ]

    dynamic_ds.close()

    if not np.array_equal(
        lat,
        dynamic_lat,
    ):
        raise RuntimeError(
            "Latitude coordinates do not match."
        )

    if not np.array_equal(
        lon,
        dynamic_lon,
    ):
        raise RuntimeError(
            "Longitude coordinates do not match."
        )

    if np.any(
        sample_indices < 0
    ) or np.any(
        sample_indices >= total_time
    ):
        raise RuntimeError(
            "Forecast sample index falls outside the dynamic dataset."
        )

    precipitation_samples = (
        precipitation_all[
            sample_indices
        ]
    )

    if precipitation_samples.shape != (
        p50.shape[0],
        p50.shape[2],
        p50.shape[3],
    ):
        raise RuntimeError(
            "Unexpected precipitation sample shape."
        )

    static_ds = xr.open_zarr(
        static_path,
        consolidated=True,
    )

    elevation = (
        static_ds[
            "elevation"
        ]
        .values
        .astype(np.float32)
    )

    river_distance = (
        static_ds[
            "river_distance_km"
        ]
        .values
        .astype(np.float32)
    )

    slope = (
        static_ds[
            "slope_degrees"
        ]
        .values
        .astype(np.float32)
    )

    river_mask = (
        static_ds[
            "river_mask"
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
            "Static latitude coordinates do not match."
        )

    if not np.array_equal(
        lon,
        static_lon,
    ):
        raise RuntimeError(
            "Static longitude coordinates do not match."
        )

    river_mask_bool = (
        river_mask > 0.5
    )

    land_mask_bool = (
        land_mask > 0.5
    )

    forecast_valid = (
        forecast_mask > 0.5
    )

    # -------------------------------------------------------------------------
    # Component 1: predicted discharge.
    #
    # Higher discharge -> higher hazard.
    # -------------------------------------------------------------------------

    discharge_component = np.zeros_like(
        p50,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Component 2: predictive uncertainty.
    #
    # Wider P10-P90 interval -> higher hazard.
    # -------------------------------------------------------------------------

    uncertainty_width = (
        p90 - p50
    )

    uncertainty_component = np.zeros_like(
        p50,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Component 3: rainfall.
    #
    # Higher rainfall -> higher hazard.
    #
    # We use the rainfall associated with each forecast sample. Because rainfall
    # here is the observed precipitation from the historical diagnostic
    # forecast, this layer is NOT an operational weather forecast yet.
    # -------------------------------------------------------------------------

    rainfall_component = np.zeros_like(
        p50,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Static components.
    # -------------------------------------------------------------------------

    static_valid = (
        land_mask_bool
        & np.isfinite(
            elevation
        )
    )

    elevation_component_2d = robust_minmax(
        elevation,
        static_valid,
    )

    elevation_component_2d = (
        inverse_robust_minmax(
            elevation_component_2d
        )
    )

    distance_valid = (
        land_mask_bool
        & np.isfinite(
            river_distance
        )
    )

    river_distance_component_2d = robust_minmax(
        river_distance,
        distance_valid,
    )

    river_distance_component_2d = (
        inverse_robust_minmax(
            river_distance_component_2d
        )
    )

    slope_valid = (
        land_mask_bool
        & np.isfinite(
            slope
        )
    )

    slope_component_2d = robust_minmax(
        slope,
        slope_valid,
    )

    slope_component_2d = (
        inverse_robust_minmax(
            slope_component_2d
        )
    )

    # -------------------------------------------------------------------------
    # Calculate dynamic components independently for every forecast lead.
    # -------------------------------------------------------------------------

    for lead in range(
        p50.shape[1]
    ):
        lead_valid = (
            forecast_valid[
                :,
                lead,
            ]
        )

        discharge_component[
            :,
            lead,
        ] = robust_minmax(
            p50[
                :,
                lead,
            ],
            lead_valid,
        )

        uncertainty_component[
            :,
            lead,
        ] = robust_minmax(
            uncertainty_width[
                :,
                lead,
            ],
            lead_valid,
        )

        rainfall_component[
            :,
            lead,
        ] = robust_minmax(
            precipitation_samples,
            np.isfinite(
                precipitation_samples
            ),
        )

    # -------------------------------------------------------------------------
    # Broadcast static components.
    # -------------------------------------------------------------------------

    elevation_component = np.broadcast_to(
        elevation_component_2d[
            None,
            None,
            ...,
        ],
        p50.shape,
    ).astype(np.float32)

    river_distance_component = np.broadcast_to(
        river_distance_component_2d[
            None,
            None,
            ...,
        ],
        p50.shape,
    ).astype(np.float32)

    slope_component = np.broadcast_to(
        slope_component_2d[
            None,
            None,
            ...,
        ],
        p50.shape,
    ).astype(np.float32)

    # -------------------------------------------------------------------------
    # Weighted hazard score.
    # -------------------------------------------------------------------------

    hazard_score = (
        weights["discharge"]
        * discharge_component
        + weights["uncertainty"]
        * uncertainty_component
        + weights["rainfall"]
        * rainfall_component
        + weights["elevation"]
        * elevation_component
        + weights["river_distance"]
        * river_distance_component
        + weights["slope"]
        * slope_component
    )

    hazard_score = np.clip(
        hazard_score,
        0.0,
        1.0,
    ).astype(np.float32)

    # Only report hazard over land and valid forecast cells.
    final_valid = (
        forecast_valid
        & land_mask_bool[
            None,
            None,
            ...,
        ]
    )

    hazard_score = np.where(
        final_valid,
        hazard_score,
        np.nan,
    )

    discharge_component = np.where(
        final_valid,
        discharge_component,
        np.nan,
    )

    uncertainty_component = np.where(
        final_valid,
        uncertainty_component,
        np.nan,
    )

    rainfall_component = np.where(
        final_valid,
        rainfall_component,
        np.nan,
    )

    elevation_component = np.where(
        final_valid,
        elevation_component,
        np.nan,
    )

    river_distance_component = np.where(
        final_valid,
        river_distance_component,
        np.nan,
    )

    slope_component = np.where(
        final_valid,
        slope_component,
        np.nan,
    )

    output = xr.Dataset(
        {
            "hydrological_hazard_score": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                hazard_score,
            ),
            "discharge_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                discharge_component,
            ),
            "uncertainty_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                uncertainty_component,
            ),
            "rainfall_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                rainfall_component,
            ),
            "elevation_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                elevation_component,
            ),
            "river_distance_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                river_distance_component,
            ),
            "slope_component": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                slope_component,
            ),
        },
        coords={
            "sample": sample,
            "forecast_day": forecast_day,
            "lat": lat,
            "lon": lon,
        },
    )

    output.attrs[
        "model"
    ] = "Bangladesh Flood World Model V2-Population"

    output.attrs[
        "hazard_type"
    ] = "heuristic_hydrological_hazard_score"

    output.attrs[
        "calibrated_probability"
    ] = "false"

    output.attrs[
        "forecast_forcing"
    ] = "historical_observed_precipitation"

    output.attrs[
        "evaluation_mask"
    ] = "glofas_discharge_valid_t × land_mask"

    output.attrs[
        "weights"
    ] = json.dumps(
        weights
    )

    output.attrs[
        "warning"
    ] = (
        "Hazard score is a heuristic decision-support feature and is not a calibrated flood probability."
    )

    return output


def summarize_hydrological_hazard(
    dataset: xr.Dataset,
) -> dict[str, Any]:
    values = (
        dataset[
            "hydrological_hazard_score"
        ]
        .values
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return {
            "finite_values": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "finite_values": int(
            finite.size
        ),
        "min": float(
            np.min(
                finite
            )
        ),
        "mean": float(
            np.mean(
                finite
            )
        ),
        "median": float(
            np.median(
                finite
            )
        ),
        "max": float(
            np.max(
                finite
            )
        ),
    }


def save_hydrological_hazard(
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

    summary = summarize_hydrological_hazard(
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
