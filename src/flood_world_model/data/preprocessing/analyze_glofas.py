from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

GLOFAS_PATH = Path(
    "data/interim/glofas/glofas_2015_2026.zarr"
)

OUTPUT_DIR = Path(
    "data/interim/glofas/analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 80)
print("GloFAS DATA QUALITY AUDIT")
print("=" * 80)

print(f"Dataset: {GLOFAS_PATH}")
print(f"Analysis output: {OUTPUT_DIR}")


# ============================================================
# 1. OPEN DATASET
# ============================================================

if not GLOFAS_PATH.exists():
    raise FileNotFoundError(
        f"GloFAS dataset not found:\n{GLOFAS_PATH}"
    )


ds = xr.open_zarr(
    GLOFAS_PATH,
    consolidated=True,
)


# ============================================================
# 2. BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("1. DATASET STRUCTURE")
print("=" * 80)

print(ds)

print("\nDimensions:")
for name, size in ds.sizes.items():
    print(
        f"  {name:15s}: {size:,}"
    )

print("\nCoordinates:")
for name, coord in ds.coords.items():
    print(
        f"  {name:15s}: "
        f"dims={coord.dims}, "
        f"shape={coord.shape}, "
        f"dtype={coord.dtype}"
    )

print("\nVariables:")
for name, da in ds.data_vars.items():
    print(
        f"  {name:25s} "
        f"dims={da.dims} "
        f"shape={da.shape} "
        f"dtype={da.dtype}"
    )


# ============================================================
# 3. COORDINATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("2. COORDINATE ANALYSIS")
print("=" * 80)


def analyze_coordinate(
    ds,
    name,
):
    if name not in ds.coords:
        print(
            f"⚠️ {name} not present"
        )
        return

    values = ds[name].values

    print(
        f"\n{name}"
    )

    print(
        f"  min: {values.min()}"
    )

    print(
        f"  max: {values.max()}"
    )

    print(
        f"  count: {len(values)}"
    )

    # Duplicate values
    unique_count = len(
        np.unique(values)
    )

    duplicate_count = (
        len(values)
        - unique_count
    )

    print(
        f"  duplicates: "
        f"{duplicate_count}"
    )

    if len(values) > 1:

        diffs = np.diff(
            values.astype(float)
        )

        print(
            f"  spacing min: "
            f"{diffs.min()}"
        )

        print(
            f"  spacing max: "
            f"{diffs.max()}"
        )

        print(
            f"  spacing median: "
            f"{np.median(diffs)}"
        )

        if np.all(diffs > 0):

            print(
                "  order: ascending ✅"
            )

        elif np.all(diffs < 0):

            print(
                "  order: descending ⚠️"
            )

        else:

            print(
                "  order: irregular ❌"
            )


analyze_coordinate(
    ds,
    "lat",
)

analyze_coordinate(
    ds,
    "lon",
)


# ============================================================
# 4. TIME ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("3. TIME ANALYSIS")
print("=" * 80)


if "time" not in ds.coords:

    raise RuntimeError(
        "GloFAS dataset has no time coordinate."
    )


time_values = pd.to_datetime(
    ds.time.values
)

print(
    f"Number of timestamps: "
    f"{len(time_values):,}"
)

print(
    f"Start: {time_values.min()}"
)

print(
    f"End: {time_values.max()}"
)


# Duplicates

duplicate_times = (
    time_values.duplicated()
)

print(
    f"Duplicate timestamps: "
    f"{duplicate_times.sum():,}"
)


# Time gaps

if len(time_values) > 1:

    time_diff = (
        time_values[1:]
        - time_values[:-1]
    )

    print("\nTime difference distribution:")

    print(
        pd.Series(
            time_diff
        )
        .value_counts()
        .head(20)
    )

    # Daily expectation
    expected = pd.Timedelta(
        days=1
    )

    non_daily = (
        time_diff != expected
    )

    print(
        f"\nNon-daily intervals: "
        f"{non_daily.sum():,}"
    )

    if non_daily.any():

        bad_indices = np.where(
            non_daily
        )[0]

        print(
            "\nFirst time gaps:"
        )

        for idx in bad_indices[:20]:

            print(
                f"  {time_values[idx]} "
                f"→ "
                f"{time_values[idx + 1]} "
                f"= "
                f"{time_diff[idx]}"
            )


# ============================================================
# 5. VARIABLE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("4. VARIABLE QUALITY")
print("=" * 80)


results = {}


for variable in ds.data_vars:

    print(
        "\n" + "-" * 70
    )

    print(
        f"Variable: {variable}"
    )

    da = ds[variable]

    result = {
        "dims": list(da.dims),
        "shape": list(da.shape),
        "dtype": str(da.dtype),
    }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print(
        "\nAttributes:"
    )

    for key, value in da.attrs.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Count missing values
    # --------------------------------------------------------

    try:

        nan_count = int(
            da.isnull()
            .sum()
            .compute()
        )

        total_count = int(
            da.size
        )

        nan_fraction = (
            nan_count
            / total_count
        )

        result[
            "nan_count"
        ] = nan_count

        result[
            "total_count"
        ] = total_count

        result[
            "nan_fraction"
        ] = nan_fraction

        print(
            f"\nNaN: "
            f"{nan_count:,} / "
            f"{total_count:,} "
            f"({nan_fraction:.2%})"
        )

    except Exception as exc:

        print(
            f"⚠️ NaN calculation failed: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Inf
    # --------------------------------------------------------

    try:

        inf_count = int(
            xr.apply_ufunc(
                np.isinf,
                da,
                dask="parallelized",
                output_dtypes=[bool],
            )
            .sum()
            .compute()
        )

        result[
            "inf_count"
        ] = inf_count

        print(
            f"Inf: {inf_count:,}"
        )

    except Exception as exc:

        print(
            f"⚠️ Inf calculation failed: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    try:

        minimum = float(
            da.min(
                skipna=True
            ).compute()
        )

        maximum = float(
            da.max(
                skipna=True
            ).compute()
        )

        mean = float(
            da.mean(
                skipna=True
            ).compute()
        )

        std = float(
            da.std(
                skipna=True
            ).compute()
        )

        result[
            "min"
        ] = minimum

        result[
            "max"
        ] = maximum

        result[
            "mean"
        ] = mean

        result[
            "std"
        ] = std

        print(
            f"Min : {minimum:.8g}"
        )

        print(
            f"Max : {maximum:.8g}"
        )

        print(
            f"Mean: {mean:.8g}"
        )

        print(
            f"Std : {std:.8g}"
        )

    except Exception as exc:

        print(
            f"⚠️ Numeric statistics failed: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    try:

        negative_count = int(
            (
                da < 0
            )
            .sum()
            .compute()
        )

        result[
            "negative_count"
        ] = negative_count

        print(
            f"Negative values: "
            f"{negative_count:,}"
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Zero values
    # --------------------------------------------------------

    try:

        zero_count = int(
            (
                da == 0
            )
            .sum()
            .compute()
        )

        result[
            "zero_count"
        ] = zero_count

        print(
            f"Zero values: "
            f"{zero_count:,}"
        )

    except Exception:
        pass

    results[
        variable
    ] = result

    gc.collect()


# ============================================================
# 6. SPATIAL VALIDITY OF RIVER DISCHARGE
# ============================================================

if "river_discharge" in ds.data_vars:

    print("\n" + "=" * 80)
    print("5. RIVER DISCHARGE SPATIAL VALIDITY")
    print("=" * 80)

    discharge = ds[
        "river_discharge"
    ]

    # Fraction of time where each cell has valid discharge.
    print(
        "Calculating valid-time fraction..."
    )

    valid_fraction = (
        discharge
        .notnull()
        .mean("time")
        .compute()
    )

    valid_values = (
        valid_fraction.values
    )

    finite = np.isfinite(
        valid_values
    )

    if finite.any():

        v = valid_values[
            finite
        ]

        print(
            f"Spatial cells: "
            f"{len(v):,}"
        )

        print(
            f"Minimum valid fraction: "
            f"{v.min():.4f}"
        )

        print(
            f"Maximum valid fraction: "
            f"{v.max():.4f}"
        )

        print(
            f"Mean valid fraction: "
            f"{v.mean():.4f}"
        )

        print(
            f"Median valid fraction: "
            f"{np.median(v):.4f}"
        )

        print(
            "\nCells by validity:"
        )

        print(
            f"  >= 99% : "
            f"{(v >= .99).sum():,}"
        )

        print(
            f"  >= 90% : "
            f"{(v >= .90).sum():,}"
        )

        print(
            f"  >= 50% : "
            f"{(v >= .50).sum():,}"
        )

        print(
            f"  > 0%   : "
            f"{(v > 0).sum():,}"
        )

        print(
            f"  = 0%   : "
            f"{(v == 0).sum():,}"
        )


    # --------------------------------------------------------
    # Save validity fraction
    # --------------------------------------------------------

    validity_path = (
        OUTPUT_DIR
        / "discharge_valid_fraction.nc"
    )

    valid_fraction.to_netcdf(
        validity_path
    )

    print(
        f"\nSaved: {validity_path}"
    )

    del valid_fraction
    gc.collect()


# ============================================================
# 7. TEMPORAL MISSINGNESS FOR DISCHARGE
# ============================================================

if "river_discharge" in ds.data_vars:

    print("\n" + "=" * 80)
    print("6. RIVER DISCHARGE TEMPORAL MISSINGNESS")
    print("=" * 80)

    discharge = ds[
        "river_discharge"
    ]

    daily_valid_fraction = (
        discharge
        .notnull()
        .mean(
            dim=[
                x
                for x in discharge.dims
                if x != "time"
            ]
        )
        .compute()
    )

    print(
        "Daily spatial validity statistics:"
    )

    values = (
        daily_valid_fraction.values
    )

    print(
        f"Min: "
        f"{np.nanmin(values):.4f}"
    )

    print(
        f"Max: "
        f"{np.nanmax(values):.4f}"
    )

    print(
        f"Mean: "
        f"{np.nanmean(values):.4f}"
    )

    # Days where less than half the spatial grid has discharge.

    problematic = (
        values < 0.5
    )

    print(
        f"Days with <50% spatial validity: "
        f"{problematic.sum():,}"
    )

    if problematic.any():

        bad_dates = time_values[
            problematic
        ]

        print(
            "\nFirst problematic dates:"
        )

        for date in bad_dates[:20]:

            print(
                f"  {date}"
            )

    del daily_valid_fraction
    gc.collect()


# ============================================================
# 8. DISCHARGE NON-NEGATIVITY
# ============================================================

if "river_discharge" in ds.data_vars:

    print("\n" + "=" * 80)
    print("7. DISCHARGE PHYSICAL CHECK")
    print("=" * 80)

    discharge = ds[
        "river_discharge"
    ]

    negative_count = int(
        (
            discharge < 0
        )
        .sum()
        .compute()
    )

    print(
        f"Negative discharge values: "
        f"{negative_count:,}"
    )

    if negative_count > 0:

        print(
            "❌ Negative discharge detected."
        )

    else:

        print(
            "✅ No negative discharge."
        )


# ============================================================
# 9. EXTREME DISCHARGE CHECK
# ============================================================

if "river_discharge" in ds.data_vars:

    print("\n" + "=" * 80)
    print("8. EXTREME DISCHARGE CHECK")
    print("=" * 80)

    discharge = ds[
        "river_discharge"
    ]

    # Quantiles computed lazily.
    quantiles = (
        discharge
        .quantile(
            [
                0.5,
                0.90,
                0.95,
                0.99,
                0.999,
            ],
            dim="time",
            skipna=True,
        )
        .compute()
    )

    print(
        quantiles
    )

    del quantiles
    gc.collect()


# ============================================================
# 10. MONTHLY COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("9. MONTHLY TEMPORAL COVERAGE")
print("=" * 80)

months = (
    time_values
    .to_period("M")
    .astype(str)
)

monthly_counts = (
    pd.Series(months)
    .value_counts()
    .sort_index()
)

print(
    monthly_counts.head(20)
)

if len(monthly_counts) > 20:
    print("...")
    print(
        monthly_counts.tail(20)
    )


# ============================================================
# 11. SAVE JSON REPORT
# ============================================================

report = {
    "dataset": str(
        GLOFAS_PATH
    ),

    "dimensions": {
        k: int(v)
        for k, v in ds.sizes.items()
    },

    "time": {
        "count": int(
            len(time_values)
        ),
        "start": str(
            time_values.min()
        ),
        "end": str(
            time_values.max()
        ),
        "duplicate_count": int(
            duplicate_times.sum()
        ),
        "non_daily_intervals": int(
            (
                time_diff
                != pd.Timedelta(days=1)
            ).sum()
        ) if len(time_values) > 1 else 0,
    },

    "variables": results,
}


report_path = (
    OUTPUT_DIR
    / "glofas_quality_report.json"
)

with report_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


print(
    f"\n✅ JSON report saved: "
    f"{report_path}"
)


# ============================================================
# 12. CLOSE
# ============================================================

ds.close()

print("\n" + "=" * 80)
print("GloFAS AUDIT COMPLETE")
print("=" * 80)

print(
    "Review the output above before modifying "
    "your training dataset."
)