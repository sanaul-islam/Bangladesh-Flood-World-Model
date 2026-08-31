from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

print("=" * 80)
print("BANGLADESH FLOOD WORLD MODEL")
print("GloFAS CLEAN PREPARATION")
print("=" * 80)

print(f"Working directory: {PROJECT_ROOT}")


# ============================================================
# INPUTS
# ============================================================

GLOFAS_PATH = Path(
    "data/interim/glofas/glofas_2015_2026.zarr"
)

IMERG_PATH = Path(
    "data/processed/nasa_imerg_compact.zarr"
)


# Final native/clean GloFAS dataset
CLEAN_NATIVE_OUTPUT = Path(
    "data/interim/glofas/glofas_clean_2015_2026.zarr"
)

# GloFAS regridded to the world-model grid
REGRID_OUTPUT = Path(
    "data/interim/glofas/glofas_on_imerg_grid_2015_2026.zarr"
)

# Validity mask on world-model grid
MASK_OUTPUT = Path(
    "data/interim/glofas/glofas_discharge_valid_mask.zarr"
)


# ============================================================
# TIME PERIOD
# ============================================================

START_DATE = "2015-01-01"
END_DATE = "2026-06-01"


# ============================================================
# CHUNKS
# ============================================================

CHUNKS_NATIVE = {
    "time": 32,
    "lat": 120,
    "lon": 90,
}

CHUNKS_REGRID = {
    "time": 32,
    "lat": 60,
    "lon": 45,
}


# ============================================================
# HELPER
# ============================================================

def remove_if_exists(path: Path) -> None:

    if path.exists():

        print(
            f"Removing old output: {path}"
        )

        shutil.rmtree(path)


def print_stats(
    name: str,
    da: xr.DataArray,
) -> None:

    print(f"\n{name}")

    total = int(
        da.size
    )

    nan_count = int(
        da.isnull()
        .sum()
        .compute()
    )

    valid_count = (
        total - nan_count
    )

    print(
        f"  total: {total:,}"
    )

    print(
        f"  valid: {valid_count:,}"
    )

    print(
        f"  NaN: {nan_count:,}"
    )

    print(
        f"  valid fraction: "
        f"{valid_count / total:.4%}"
    )

    finite = (
        da.isnull()
        .compute()
        .values
    )


# ============================================================
# CHECK INPUT
# ============================================================

if not GLOFAS_PATH.exists():
    raise FileNotFoundError(
        f"GloFAS not found:\n{GLOFAS_PATH}"
    )

if not IMERG_PATH.exists():
    raise FileNotFoundError(
        f"IMERG not found:\n{IMERG_PATH}"
    )


# ============================================================
# LOAD IMERG GRID
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING IMERG MODEL GRID")
print("=" * 80)

imerg = xr.open_zarr(
    IMERG_PATH,
    consolidated=True,
)

target_lat = np.asarray(
    imerg.lat.values,
    dtype=np.float64,
)

target_lon = np.asarray(
    imerg.lon.values,
    dtype=np.float64,
)

target_lat = np.sort(target_lat)
target_lon = np.sort(target_lon)

print(
    f"IMERG grid: "
    f"{len(target_lat)} × "
    f"{len(target_lon)}"
)

print(
    f"Lat: "
    f"{target_lat.min()} → "
    f"{target_lat.max()}"
)

print(
    f"Lon: "
    f"{target_lon.min()} → "
    f"{target_lon.max()}"
)


# ============================================================
# LOAD GLOFAS
# ============================================================

print("\n" + "=" * 80)
print("2. LOADING GLOFAS")
print("=" * 80)

glofas = xr.open_zarr(
    GLOFAS_PATH,
    consolidated=True,
)

print(
    "Original GloFAS:",
    dict(glofas.sizes),
)

print(
    "Original time:",
    glofas.time.min().values,
    "→",
    glofas.time.max().values,
)


# ============================================================
# VALIDATE VARIABLE
# ============================================================

if "river_discharge" not in glofas.data_vars:

    raise ValueError(
        "river_discharge variable not found."
    )


# ============================================================
# STANDARDIZE COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("3. STANDARDIZING COORDINATES")
print("=" * 80)

# Sort lat/lon.
#
# GloFAS currently has descending latitude.
# We convert it to ascending.

glofas = glofas.sortby("lat")
glofas = glofas.sortby("lon")
glofas = glofas.sortby("time")


# ============================================================
# REMOVE DUPLICATE TIMES
# ============================================================

time_values = glofas.time.values

_, unique_idx = np.unique(
    time_values,
    return_index=True,
)

unique_idx = np.sort(
    unique_idx
)

if len(unique_idx) != len(time_values):

    print(
        "Removing duplicate timestamps..."
    )

    glofas = glofas.isel(
        time=unique_idx
    )


# ============================================================
# SELECT PROJECT PERIOD
# ============================================================

print("\n" + "=" * 80)
print("4. SELECTING PROJECT PERIOD")
print("=" * 80)

glofas = glofas.sel(
    time=slice(
        START_DATE,
        END_DATE,
    )
)

print(
    f"Selected period: "
    f"{glofas.time.min().values} → "
    f"{glofas.time.max().values}"
)

print(
    f"Days: "
    f"{glofas.sizes['time']:,}"
)


# ============================================================
# KEEP ONLY DISCHARGE
# ============================================================

glofas = glofas[
    ["river_discharge"]
].copy()


# ============================================================
# CAST
# ============================================================

glofas[
    "river_discharge"
] = (
    glofas[
        "river_discharge"
    ]
    .astype("float32")
)


# ============================================================
# REMOVE INFINITIES
# ============================================================

glofas[
    "river_discharge"
] = (
    glofas[
        "river_discharge"
    ].where(
        np.isfinite(
            glofas[
                "river_discharge"
            ]
        )
    )
)


# ============================================================
# CHECK NEGATIVE VALUES
# ============================================================

negative_count = int(
    (
        glofas[
            "river_discharge"
        ]
        < 0
    )
    .sum()
    .compute()
)

print(
    f"Negative discharge: "
    f"{negative_count:,}"
)

if negative_count > 0:

    raise RuntimeError(
        "Negative discharge detected."
    )


# ============================================================
# PRINT NATIVE STATS
# ============================================================

print_stats(
    "Native GloFAS discharge",
    glofas[
        "river_discharge"
    ],
)


# ============================================================
# SAVE CLEAN NATIVE GLOFAS
# ============================================================

remove_if_exists(
    CLEAN_NATIVE_OUTPUT
)

print("\nSaving cleaned native GloFAS...")

glofas = glofas.chunk(
    CHUNKS_NATIVE
)

glofas.to_zarr(
    CLEAN_NATIVE_OUTPUT,
    mode="w",
    consolidated=True,
)

print(
    f"✅ Saved: "
    f"{CLEAN_NATIVE_OUTPUT}"
)


# ============================================================
# CREATE STATIC/LONG-TERM VALIDITY MASK
# ============================================================

print("\n" + "=" * 80)
print("5. BUILDING NATIVE VALIDITY MASK")
print("=" * 80)

#
# A cell is considered structurally valid if it has
# discharge data on at least one day.
#
# In your current audit this should produce:
#
# ~8,481 valid cells
# ~2,319 permanently invalid cells
#

native_valid_mask = (
    glofas[
        "river_discharge"
    ]
    .notnull()
    .any(
        dim="time"
    )
    .astype("float32")
)

native_valid_mask.name = (
    "glofas_discharge_valid_mask"
)

native_valid_mask.attrs = {
    "description": (
        "1 if GloFAS provides discharge "
        "for this spatial cell during "
        "the selected period; 0 otherwise."
    ),
    "role": "target_validity_mask",
}


native_valid_mask = (
    native_valid_mask.compute()
)


print(
    f"Native valid cells: "
    f"{int(native_valid_mask.sum())}"
)

print(
    f"Native total cells: "
    f"{native_valid_mask.size}"
)


# ============================================================
# REGRID DISCHARGE TO IMERG GRID
# ============================================================

print("\n" + "=" * 80)
print("6. REGRIDDING GloFAS → IMERG")
print("=" * 80)

print(
    "Using NEAREST NEIGHBOR."
)

print(
    "This is intentional because river discharge "
    "is a network/sparse variable."
)


glofas_on_imerg = glofas.interp(
    lat=target_lat,
    lon=target_lon,
    method="nearest",
)


# ============================================================
# REGRID VALIDITY MASK
# ============================================================

print(
    "\nRegridding validity mask..."
)

valid_mask_on_imerg = (
    native_valid_mask.interp(
        lat=target_lat,
        lon=target_lon,
        method="nearest",
    )
)


valid_mask_on_imerg = (
    valid_mask_on_imerg
    .astype("float32")
    .rename(
        "glofas_discharge_valid_mask"
    )
)


# ============================================================
# CREATE TIME-VARYING VALIDITY MASK
# ============================================================

print(
    "\nCreating time-varying valid mask..."
)

time_valid_mask = (
    glofas_on_imerg[
        "river_discharge"
    ]
    .notnull()
    .astype("float32")
    .rename(
        "glofas_discharge_valid_t"
    )
)


# ============================================================
# ADD MASKS TO REGRIDDED DATA
# ============================================================

glofas_on_imerg[
    "glofas_discharge_valid_mask"
] = (
    valid_mask_on_imerg
)

glofas_on_imerg[
    "glofas_discharge_valid_t"
] = (
    time_valid_mask
)


# ============================================================
# METADATA
# ============================================================

glofas_on_imerg.attrs = {
    "title": (
        "GloFAS discharge aligned to "
        "NASA IMERG model grid"
    ),
    "source": "GloFAS",
    "target_grid": "NASA IMERG 0.1 degree",
    "regridding": "nearest neighbor",
    "period": (
        f"{START_DATE} → {END_DATE}"
    ),
}


glofas_on_imerg[
    "river_discharge"
].attrs.update(
    {
        "source": "GloFAS",
        "units": "m3 s-1",
        "role": "hydrological_state",
        "regridding": "nearest neighbor",
    }
)


# ============================================================
# CAST / CHUNK
# ============================================================

for variable in glofas_on_imerg.data_vars:

    glofas_on_imerg[
        variable
    ] = (
        glofas_on_imerg[
            variable
        ]
        .astype("float32")
    )


glofas_on_imerg = (
    glofas_on_imerg.chunk(
        CHUNKS_REGRID
    )
)


# ============================================================
# REMOVE OLD REGRIDDED DATA
# ============================================================

remove_if_exists(
    REGRID_OUTPUT
)


# ============================================================
# SAVE REGRIDDED DATA
# ============================================================

print("\n" + "=" * 80)
print("7. SAVING GloFAS ON MODEL GRID")
print("=" * 80)

glofas_on_imerg.to_zarr(
    REGRID_OUTPUT,
    mode="w",
    consolidated=True,
)

print(
    f"✅ Saved: {REGRID_OUTPUT}"
)


# ============================================================
# SAVE SEPARATE MASK
# ============================================================

remove_if_exists(
    MASK_OUTPUT
)

mask_dataset = xr.Dataset(
    {
        "glofas_discharge_valid_mask":
            valid_mask_on_imerg,
    }
)

mask_dataset.attrs = {
    "description": (
        "Static spatial mask indicating "
        "whether a GloFAS discharge cell "
        "exists in the selected project period."
    ),
    "value_0": "No discharge field",
    "value_1": "Discharge field available",
}

mask_dataset = (
    mask_dataset
    .chunk(
        {
            "lat": len(target_lat),
            "lon": len(target_lon),
        }
    )
)

mask_dataset.to_zarr(
    MASK_OUTPUT,
    mode="w",
    consolidated=True,
)

print(
    f"✅ Saved: {MASK_OUTPUT}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("8. FINAL VERIFICATION")
print("=" * 80)

check = xr.open_zarr(
    REGRID_OUTPUT,
    consolidated=True,
)

print(
    check
)

print(
    "\nDimensions:",
    dict(check.sizes),
)

print(
    "\nVariables:"
)

for variable in check.data_vars:

    print(
        f"  ✅ {variable:35s}"
        f"{check[variable].shape}"
        f" {check[variable].dtype}"
    )


# ------------------------------------------------------------
# Validity statistics
# ------------------------------------------------------------

valid_mask = check[
    "glofas_discharge_valid_mask"
]

valid_cells = int(
    valid_mask.sum()
    .compute()
)

total_cells = int(
    valid_mask.size
)

print(
    f"\nValid cells on IMERG grid: "
    f"{valid_cells}/{total_cells}"
)

print(
    f"Valid fraction: "
    f"{valid_cells / total_cells:.2%}"
)


# ------------------------------------------------------------
# Actual discharge missingness
# ------------------------------------------------------------

discharge = check[
    "river_discharge"
]

overall_missing = int(
    discharge.isnull()
    .sum()
    .compute()
)

overall_total = int(
    discharge.size
)

print(
    f"\nRegridded discharge missing:"
)

print(
    f"  {overall_missing:,} / "
    f"{overall_total:,} "
    f"({overall_missing / overall_total:.2%})"
)


# ------------------------------------------------------------
# Negative values
# ------------------------------------------------------------

negative = int(
    (
        discharge < 0
    )
    .sum()
    .compute()
)

print(
    f"Negative values: {negative}"
)


# ============================================================
# CLOSE EVERYTHING
# ============================================================

check.close()

glofas.close()

imerg.close()

mask_dataset.close()

gc.collect()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("🎉 GloFAS PREPARATION COMPLETE")
print("=" * 80)

print(
    "\nCreated:"
)

print(
    f"  ✅ {CLEAN_NATIVE_OUTPUT}"
)

print(
    f"  ✅ {REGRID_OUTPUT}"
)

print(
    f"  ✅ {MASK_OUTPUT}"
)

print(
    "\nImportant:"
)

print(
    "  NaN discharge cells were NOT replaced by zero."
)

print(
    "  Discharge was regridded using nearest neighbor."
)

print(
    "  A structural validity mask was created."
)

print(
    "\nNext:"
)

print(
    "Use the validity mask when calculating "
    "the world-model discharge loss."
)