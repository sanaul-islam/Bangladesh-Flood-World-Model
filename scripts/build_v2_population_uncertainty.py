from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/v2_vs_population_test.nc"
)

OUTPUT_FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/v2_population_uncertainty.nc"
)

OUTPUT_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/v2_population_uncertainty.json"
)

HORIZON = 7


def calculate_interval_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
) -> dict:
    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & np.isfinite(p10)
        & np.isfinite(p90)
    )

    if not valid.any():
        return {
            "valid_values": 0,
            "coverage_10_90": None,
            "mean_interval_width_m3_s": None,
            "mean_abs_error_m3_s": None,
        }

    y = actual[valid]
    p = predicted[valid]
    lower = p10[valid]
    upper = p90[valid]

    covered = (
        (y >= lower)
        & (y <= upper)
    )

    width = upper - lower

    absolute_error = np.abs(
        p - y
    )

    return {
        "valid_values": int(
            valid.sum()
        ),
        "coverage_10_90": float(
            covered.mean()
        ),
        "mean_interval_width_m3_s": float(
            width.mean()
        ),
        "mean_abs_error_m3_s": float(
            absolute_error.mean()
        ),
    }


def main() -> None:
    print("=" * 80)
    print("V2-POPULATION UNCERTAINTY ESTIMATION")
    print("=" * 80)

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Missing forecast file: {FORECAST_PATH}"
        )

    ds = xr.open_dataset(
        FORECAST_PATH
    )

    prediction = (
        ds[
            "v2_population_predicted_discharge"
        ]
        .values
        .astype(np.float32)
    )

    actual = (
        ds[
            "actual_discharge"
        ]
        .values
        .astype(np.float32)
    )

    mask = (
        ds[
            "evaluation_mask"
        ]
        .values
        .astype(np.float32)
    )

    lat = ds.lat.values
    lon = ds.lon.values
    sample = ds.sample.values
    forecast_day = ds.forecast_day.values

    ds.close()

    if prediction.ndim != 4:
        raise RuntimeError(
            f"Expected prediction shape [sample, day, lat, lon], got {prediction.shape}"
        )

    if prediction.shape[1] != HORIZON:
        raise RuntimeError(
            f"Expected {HORIZON} forecast days, got {prediction.shape[1]}"
        )

    if actual.shape != prediction.shape:
        raise RuntimeError(
            "Prediction and actual shapes differ."
        )

    if mask.shape != prediction.shape:
        raise RuntimeError(
            "Prediction and mask shapes differ."
        )

    # -------------------------------------------------------------------------
    # Residuals.
    #
    # residual = actual - prediction
    #
    # A positive residual means the model underpredicted.
    # A negative residual means the model overpredicted.
    #
    # Residual quantiles are calculated independently for each forecast lead.
    # -------------------------------------------------------------------------

    p10 = np.full_like(
        prediction,
        np.nan,
    )

    p50 = np.full_like(
        prediction,
        np.nan,
    )

    p90 = np.full_like(
        prediction,
        np.nan,
    )

    residual_metrics = []

    for lead in range(
        HORIZON
    ):
        lead_prediction = prediction[
            :,
            lead,
            ...
        ]

        lead_actual = actual[
            :,
            lead,
            ...
        ]

        lead_mask = (
            mask[
                :,
                lead,
                ...
            ]
            > 0.5
        )

        residual = (
            lead_actual
            - lead_prediction
        )

        valid = (
            lead_mask
            & np.isfinite(
                residual
            )
        )

        if not valid.any():
            raise RuntimeError(
                f"No valid residuals for lead day {lead + 1}."
            )

        values = residual[
            valid
        ].astype(
            np.float64
        )

        lower_error = float(
            np.percentile(
                values,
                10,
            )
        )

        median_error = float(
            np.percentile(
                values,
                50,
            )
        )

        upper_error = float(
            np.percentile(
                values,
                90,
            )
        )

        p10[
            :,
            lead,
            ...
        ] = (
            lead_prediction
            + lower_error
        )

        p50[
            :,
            lead,
            ...
        ] = (
            lead_prediction
            + median_error
        )

        p90[
            :,
            lead,
            ...
        ] = (
            lead_prediction
            + upper_error
        )

        p10[
            :,
            lead,
            ...
        ] = np.maximum(
            p10[
                :,
                lead,
                ...
            ],
            0.0,
        )

        p50[
            :,
            lead,
            ...
        ] = np.maximum(
            p50[
                :,
                lead,
                ...
            ],
            0.0,
        )

        p90[
            :,
            lead,
            ...
        ] = np.maximum(
            p90[
                :,
                lead,
                ...
            ],
            0.0,
        )

        metrics = calculate_interval_metrics(
            actual=lead_actual[lead_mask],
            predicted=lead_prediction[lead_mask],
            p10=p10[:, lead, ...][lead_mask],
            p90=p90[:, lead, ...][lead_mask],
        )

        metrics.update(
            {
                "lead_day": lead + 1,
                "residual_p10_m3_s": lower_error,
                "residual_p50_m3_s": median_error,
                "residual_p90_m3_s": upper_error,
            }
        )

        residual_metrics.append(
            metrics
        )

        print("-" * 80)
        print(
            f"DAY {lead + 1}"
        )
        print(
            f"Residual P10: {lower_error:.3f} m3/s"
        )
        print(
            f"Residual P50: {median_error:.3f} m3/s"
        )
        print(
            f"Residual P90: {upper_error:.3f} m3/s"
        )
        print(
            f"80% interval coverage: {metrics['coverage_10_90'] * 100.0:.2f}%"
        )
        print(
            f"Mean interval width: {metrics['mean_interval_width_m3_s']:.3f} m3/s"
        )

    output_ds = xr.Dataset(
        {
            "predicted_discharge_p50": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                p50.astype(np.float32),
            ),
            "predicted_discharge_p10": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                p10.astype(np.float32),
            ),
            "predicted_discharge_p90": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                p90.astype(np.float32),
            ),
            "evaluation_mask": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                mask.astype(np.float32),
            ),
        },
        coords={
            "sample": sample,
            "forecast_day": forecast_day,
            "lat": lat,
            "lon": lon,
        },
    )

    output_ds[
        "predicted_discharge_p10"
    ].attrs["units"] = "m3 s-1"

    output_ds[
        "predicted_discharge_p50"
    ].attrs["units"] = "m3 s-1"

    output_ds[
        "predicted_discharge_p90"
    ].attrs["units"] = "m3 s-1"

    output_ds.attrs[
        "model"
    ] = "Bangladesh Flood World Model V2-Population"

    output_ds.attrs[
        "uncertainty_method"
    ] = (
        "lead-specific empirical residual quantiles"
    )

    output_ds.attrs[
        "interval"
    ] = "P10-P90"

    output_ds.attrs[
        "interpretation"
    ] = (
        "Empirical predictive interval, not a calibrated probability distribution."
    )

    OUTPUT_FORECAST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_ds.to_netcdf(
        OUTPUT_FORECAST_PATH
    )

    output_ds.close()

    result = {
        "model": "Bangladesh Flood World Model V2-Population",
        "method": "lead_specific_empirical_residual_quantiles",
        "interval": "P10-P90",
        "calibrated": False,
        "source_forecast": str(
            FORECAST_PATH
        ),
        "output_forecast": str(
            OUTPUT_FORECAST_PATH
        ),
        "lead_metrics": residual_metrics,
    }

    OUTPUT_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("UNCERTAINTY ESTIMATION COMPLETE")
    print("=" * 80)

    print(
        f"Forecast: {OUTPUT_FORECAST_PATH}"
    )

    print(
        f"Metrics: {OUTPUT_METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
