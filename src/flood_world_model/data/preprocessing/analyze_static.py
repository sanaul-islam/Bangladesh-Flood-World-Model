from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

STATIC_PATH = Path(
    "data/features/static.zarr"
)

OUTPUT_DIR = Path(
    "data/features/analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OPEN
# ============================================================

if not STATIC_PATH.exists():
    raise FileNotFoundError(
        f"Static dataset not found:\n{STATIC_PATH}"
    )

ds = xr.open_zarr(
    STATIC_PATH,
    consolidated=True,
)

print("=" * 80)
print("STATIC DATA QUALITY AUDIT")
print("=" * 80)

print("\nDataset:")
print(ds)

print("\nDimensions:")
for name, size in ds.sizes.items():
    print(f"  {name:25s}: {size}")

print("\nVariables:")
for name in ds.data_vars:
    print(
        f"  {name:25s} "
        f"{ds[name].shape} "
        f"{ds[name].dtype}"
    )


# ============================================================
# GRID
# ============================================================

print("\n" + "=" * 80)
print("GRID")
print("=" * 80)

lat = ds.lat.values
lon = ds.lon.values

print(
    f"Latitude: {lat.min()} → {lat.max()}"
)

print(
    f"Longitude: {lon.min()} → {lon.max()}"
)

print(
    f"Grid: {len(lat)} × {len(lon)}"
)


# ============================================================
# LAND MASK
# ============================================================

if "elevation" not in ds:

    raise RuntimeError(
        "Elevation is required to construct the land mask."
    )

land_mask = np.isfinite(
    ds["elevation"].values
)

print("\n" + "=" * 80)
print("LAND MASK")
print("=" * 80)

land_cells = int(
    land_mask.sum()
)

total_cells = int(
    land_mask.size
)

print(
    f"Land/valid DEM cells: "
    f"{land_cells:,}/{total_cells:,}"
)

print(
    f"Land fraction: "
    f"{land_cells / total_cells:.2%}"
)


# ============================================================
# PER-VARIABLE ANALYSIS
# ============================================================

report = {}

print("\n" + "=" * 80)
print("VARIABLE ANALYSIS")
print("=" * 80)

for name in ds.data_vars:

    da = ds[name]

    values = da.values

    total = values.size

    finite = np.isfinite(
        values
    )

    valid_count = int(
        finite.sum()
    )

    nan_count = int(
        np.isnan(values).sum()
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    result = {
        "shape": list(da.shape),
        "dtype": str(da.dtype),
        "total": total,
        "finite": valid_count,
        "nan": nan_count,
        "inf": inf_count,
        "valid_fraction": (
            valid_count / total
            if total
            else 0.0
        ),
    }

    print(
        f"\n{name}"
    )

    print(
        f"  valid: "
        f"{valid_count:,}/{total:,} "
        f"({valid_count / total:.2%})"
    )

    print(
        f"  NaN: {nan_count:,}"
    )

    print(
        f"  Inf: {inf_count:,}"
    )

    if valid_count:

        valid_values = values[
            finite
        ]

        result["min"] = float(
            valid_values.min()
        )

        result["max"] = float(
            valid_values.max()
        )

        result["mean"] = float(
            valid_values.mean()
        )

        result["std"] = float(
            valid_values.std()
        )

        print(
            f"  min: {result['min']}"
        )

        print(
            f"  max: {result['max']}"
        )

        print(
            f"  mean: {result['mean']}"
        )

        print(
            f"  std: {result['std']}"
        )


    # --------------------------------------------------------
    # NaNs on land
    # --------------------------------------------------------

    if values.ndim == 2:

        nan_on_land = (
            (~finite)
            & land_mask
        )

        nan_on_nonland = (
            (~finite)
            & (~land_mask)
        )

        result["nan_on_land"] = int(
            nan_on_land.sum()
        )

        result["nan_on_nonland"] = int(
            nan_on_nonland.sum()
        )

        print(
            f"  NaN on land: "
            f"{int(nan_on_land.sum()):,}"
        )

        print(
            f"  NaN outside land: "
            f"{int(nan_on_nonland.sum()):,}"
        )


    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    negative_count = int(
        (
            values < 0
        )
        .sum()
    )

    result["negative"] = negative_count

    if negative_count:
        print(
            f"  ⚠️ negative values: "
            f"{negative_count:,}"
        )
    else:
        print(
            "  negative: 0"
        )


    report[name] = result


# ============================================================
# LAND COVER CLASSES
# ============================================================

if "landcover" in ds:

    print("\n" + "=" * 80)
    print("LAND COVER CLASSES")
    print("=" * 80)

    values = ds[
        "landcover"
    ].values

    valid = np.isfinite(
        values
    )

    unique, counts = np.unique(
        values[valid],
        return_counts=True,
    )

    for value, count in zip(
        unique,
        counts,
    ):

        print(
            f"  class {value:g}: "
            f"{count:,} cells"
        )


# ============================================================
# RIVER MASK
# ============================================================

if "river_mask" in ds:

    print("\n" + "=" * 80)
    print("RIVER MASK")
    print("=" * 80)

    mask = ds[
        "river_mask"
    ].values

    print(
        f"River cells: "
        f"{int((mask > 0.5).sum()):,}"
    )

    print(
        f"Non-river cells: "
        f"{int((mask <= 0.5).sum()):,}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report[
    "land_cells"
] = land_cells

report[
    "total_cells"
] = total_cells

report[
    "land_fraction"
] = (
    land_cells
    / total_cells
)

report_path = (
    OUTPUT_DIR
    / "static_quality_report.json"
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
    f"\n✅ Report saved: "
    f"{report_path}"
)

ds.close()

print("\n✅ Static audit complete.")