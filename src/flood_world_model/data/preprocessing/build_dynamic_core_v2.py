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
print("DYNAMIC CORE V2")
print("=" * 80)


# ============================================================
# INPUTS
# ============================================================

IMERG_PATH = Path(
    "data/processed/nasa_imerg_compact.zarr"
)

GLOFAS_PATH = Path(
    "data/interim/glofas/"
    "glofas_on_imerg_grid_2015_2026.zarr"
)

OUTPUT_PATH = Path(
    "data/features/dynamic_core_v2.zarr"
)


# ============================================================
# CONFIG
# ============================================================

# Use IMERG as canonical time/grid source.
#
# Only use the period where both datasets actually exist.

START_DATE = "2015-01-01"
END_DATE = "2026-06-01"


# Training-friendly chunks.
CHUNKS = {
    "time": 32,
    "lat": 60,
    "lon": 45,
}


# ============================================================
# CHECK INPUTS
# ============================================================

if not IMERG_PATH.exists():
    raise FileNotFoundError(
        f"IMERG not found:\n{IMERG_PATH}"
    )

if not GLOFAS_PATH.exists():
    raise FileNotFoundError(
        f"GloFAS not found:\n{GLOFAS_PATH}"
    )


# ============================================================
# LOAD IMERG
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING IMERG")
print("=" * 80)

imerg = xr.open_zarr(
    IMERG_PATH,
    consolidated=True,
)

if "precipitation" not in imerg.data_vars:
    raise ValueError(
        "IMERG precipitation variable not found."
    )

imerg = (
    imerg
    .sortby("time")
    .sortby("lat")
    .sortby("lon")
)

# Only precipitation.
imerg = imerg[
    ["precipitation"]
]

# Project period.
imerg = imerg.sel(
    time=slice(
        START_DATE,
        END_DATE,
    )
)

print(
    "IMERG:",
    dict(imerg.sizes)
)

print(
    "Time:",
    imerg.time.min().values,
    "→",
    imerg.time.max().values,
)


# ============================================================
# LOAD GLOFAS
# ============================================================

print("\n" + "=" * 80)
print("2. LOADING CLEAN GLOFAS")
print("=" * 80)

glofas = xr.open_zarr(
    GLOFAS_PATH,
    consolidated=True,
)

if "river_discharge" not in glofas.data_vars:

    raise ValueError(
        "Clean GloFAS dataset does not contain "
        "'river_discharge'."
    )

glofas = (
    glofas
    .sortby("time")
    .sortby("lat")
    .sortby("lon")
)

glofas = glofas.sel(
    time=slice(
        START_DATE,
        END_DATE,
    )
)

print(
    "GloFAS:",
    dict(glofas.sizes)
)

print(
    "Time:",
    glofas.time.min().values,
    "→",
    glofas.time.max().values,
)

print(
    "Variables:",
    list(glofas.data_vars)
)


# ============================================================
# GRID CHECK
# ============================================================

print("\n" + "=" * 80)
print("3. GRID CHECK")
print("=" * 80)

if not np.allclose(
    imerg.lat.values,
    glofas.lat.values,
):

    raise ValueError(
        "IMERG and GloFAS latitude grids differ."
    )

if not np.allclose(
    imerg.lon.values,
    glofas.lon.values,
):

    raise ValueError(
        "IMERG and GloFAS longitude grids differ."
    )

print(
    "✅ Latitude grids match."
)

print(
    "✅ Longitude grids match."
)


# ============================================================
# TIME ALIGNMENT
# ============================================================

print("\n" + "=" * 80)
print("4. TIME ALIGNMENT")
print("=" * 80)

imerg, glofas = xr.align(
    imerg,
    glofas,
    join="inner",
)

print(
    f"Common days: "
    f"{imerg.sizes['time']:,}"
)

print(
    "Common period:",
    imerg.time.min().values,
    "→",
    imerg.time.max().values,
)

if imerg.sizes["time"] == 0:

    raise RuntimeError(
        "No common dates between IMERG and GloFAS."
    )


# ============================================================
# BUILD CORE
# ============================================================

print("\n" + "=" * 80)
print("5. BUILDING DYNAMIC CORE")
print("=" * 80)

dynamic = xr.Dataset(
    {
        "precipitation":
            imerg["precipitation"],

        "river_discharge":
            glofas["river_discharge"],
    },
    coords={
        "time": imerg.time,
        "lat": imerg.lat,
        "lon": imerg.lon,
    },
)


# ============================================================
# CLEAN PRECIPITATION
# ============================================================

precipitation = (
    dynamic["precipitation"]
    .astype("float32")
)

# Prevent negative precipitation due to numerical artifacts.
precipitation = xr.where(
    precipitation < 0,
    0,
    precipitation,
)

dynamic[
    "precipitation"
] = precipitation


# ============================================================
# DISCHARGE
# ============================================================

discharge = (
    dynamic["river_discharge"]
    .astype("float32")
)

# Negative discharge is physically invalid.
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

    raise RuntimeError(
        "Negative discharge detected."
    )

dynamic[
    "river_discharge"
] = discharge


# ============================================================
# 6. PRECIPITATION DERIVED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("6. DERIVING PRECIPITATION FEATURES")
print("=" * 80)


# ------------------------------------------------------------
# Trailing 3-day accumulation.
#
# At time t:
# t + t-1 + t-2
#
# No future information.
# ------------------------------------------------------------

dynamic[
    "precip_3d"
] = (
    precipitation
    .rolling(
        time=3,
        min_periods=1,
    )
    .sum()
    .astype("float32")
)


# ------------------------------------------------------------
# Trailing 7-day accumulation.
# ------------------------------------------------------------

dynamic[
    "precip_7d"
] = (
    precipitation
    .rolling(
        time=7,
        min_periods=1,
    )
    .sum()
    .astype("float32")
)


# ------------------------------------------------------------
# Log transform.
# ------------------------------------------------------------

dynamic[
    "precip_log1p"
] = (
    np.log1p(
        precipitation
    )
    .astype("float32")
)


# ------------------------------------------------------------
# Rainfall missing flag.
# ------------------------------------------------------------

dynamic[
    "precip_missing"
] = (
    precipitation
    .isnull()
    .astype("float32")
)


# ============================================================
# 7. GloFAS DISCHARGE VALIDITY MASK
# ============================================================

print("\n" + "=" * 80)
print("7. GloFAS DISCHARGE VALIDITY")
print("=" * 80)

discharge_valid = (
    discharge
    .notnull()
    .astype("float32")
)

dynamic[
    "glofas_discharge_valid_t"
] = discharge_valid


# ============================================================
# 8. CLEAN NUMERIC TYPES
# ============================================================

print("\n" + "=" * 80)
print("8. FINAL TYPE CLEANUP")
print("=" * 80)

for variable in dynamic.data_vars:

    dynamic[
        variable
    ] = (
        dynamic[
            variable
        ]
        .astype("float32")
    )


# ============================================================
# 9. VARIABLE ORDER
# ============================================================

variable_order = [
    "precipitation",
    "precip_3d",
    "precip_7d",
    "precip_log1p",
    "precip_missing",
    "river_discharge",
    "glofas_discharge_valid_t",
]

dynamic = dynamic[
    variable_order
]


# ============================================================
# 10. METADATA
# ============================================================

dynamic.attrs = {

    "title":
        "Bangladesh Flood World Model Dynamic Core V2",

    "description":
        (
            "Daily rainfall and river-discharge state "
            "variables on the NASA IMERG 0.1-degree grid."
        ),

    "canonical_grid":
        "NASA IMERG 0.1 degree",

    "time_frequency":
        "daily",

    "period":
        f"{START_DATE} → {END_DATE}",

    "glofas_regridding":
        "nearest neighbor",

    "note":
        (
            "GloFAS discharge remains sparse where "
            "the source does not define discharge."
        ),
}


# ============================================================
# VARIABLE METADATA
# ============================================================

dynamic[
    "precipitation"
].attrs = {
    "source": "NASA IMERG",
    "role": "dynamic_forcing",
    "units": "source_dataset_units",
}


dynamic[
    "precip_3d"
].attrs = {
    "role": "derived_dynamic_feature",
    "description":
        "Trailing 3-day precipitation accumulation.",
}


dynamic[
    "precip_7d"
].attrs = {
    "role": "derived_dynamic_feature",
    "description":
        "Trailing 7-day precipitation accumulation.",
}


dynamic[
    "precip_log1p"
].attrs = {
    "role": "derived_dynamic_feature",
    "description":
        "log1p transformed precipitation.",
}


dynamic[
    "precip_missing"
].attrs = {
    "role": "data_quality_feature",
    "description":
        "1 where precipitation is missing, otherwise 0.",
}


dynamic[
    "river_discharge"
].attrs = {
    "source": "GloFAS",
    "role": "hydrological_state",
    "units": "m3 s-1",
}


dynamic[
    "glofas_discharge_valid_t"
].attrs = {
    "source": "GloFAS",
    "role": "target_validity_mask",
    "description":
        (
            "1 where discharge is valid at this time "
            "and location; 0 otherwise."
        ),
}


# ============================================================
# 11. VALIDATE TIME
# ============================================================

print("\n" + "=" * 80)
print("9. TIME VALIDATION")
print("=" * 80)

times = dynamic.time.values.astype(
    "datetime64[D]"
)

if len(times) > 1:

    time_diff = np.diff(
        times
    )

    bad = (
        time_diff
        != np.timedelta64(
            1,
            "D",
        )
    )

    print(
        f"Non-daily intervals: "
        f"{int(bad.sum())}"
    )

    if bad.any():

        print(
            "⚠️ There are gaps in the final dynamic cube."
        )

    else:

        print(
            "✅ Daily sequence is continuous."
        )


# ============================================================
# 12. VALIDATE GRID
# ============================================================

print("\n" + "=" * 80)
print("10. GRID VALIDATION")
print("=" * 80)

expected_shape = (
    len(times),
    len(imerg.lat),
    len(imerg.lon),
)

for variable in dynamic.data_vars:

    if dynamic[
        variable
    ].shape != expected_shape:

        raise ValueError(
            f"{variable} shape is "
            f"{dynamic[variable].shape}, "
            f"expected {expected_shape}"
        )

print(
    f"✅ Shape: {expected_shape}"
)


# ============================================================
# 13. BASIC DATA QUALITY
# ============================================================

print("\n" + "=" * 80)
print("11. DATA QUALITY")
print("=" * 80)

for variable in dynamic.data_vars:

    da = dynamic[
        variable
    ]

    nan_count = int(
        da.isnull()
        .sum()
        .compute()
    )

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

    print(
        f"{variable:30s}"
        f"NaN={nan_count:,}"
        f"  Inf={inf_count:,}"
    )


# ============================================================
# 14. DISCHARGE VALIDITY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("12. DISCHARGE VALIDITY SUMMARY")
print("=" * 80)

valid = (
    dynamic[
        "glofas_discharge_valid_t"
    ]
)

valid_fraction = float(
    valid.mean()
    .compute()
)

print(
    f"Overall discharge valid fraction: "
    f"{valid_fraction:.4%}"
)


# Spatial long-term validity.
spatial_fraction = (
    valid
    .mean("time")
    .compute()
)

print(
    f"Minimum cell validity: "
    f"{float(spatial_fraction.min()):.4f}"
)

print(
    f"Maximum cell validity: "
    f"{float(spatial_fraction.max()):.4f}"
)

print(
    f"Median cell validity: "
    f"{float(spatial_fraction.median()):.4f}"
)


# ============================================================
# 15. CHUNKING
# ============================================================

print("\n" + "=" * 80)
print("13. CHUNKING")
print("=" * 80)

print(
    CHUNKS
)

dynamic = dynamic.chunk(
    CHUNKS
)


# ============================================================
# 16. REMOVE EXISTING OUTPUT
# ============================================================

if OUTPUT_PATH.exists():

    print(
        f"\nRemoving existing:"
        f"\n{OUTPUT_PATH}"
    )

    shutil.rmtree(
        OUTPUT_PATH
    )


# ============================================================
# 17. SAVE
# ============================================================

print("\n" + "=" * 80)
print("14. WRITING DYNAMIC CORE V2")
print("=" * 80)

dynamic.to_zarr(
    OUTPUT_PATH,
    mode="w",
    consolidated=True,
)

print(
    f"✅ Saved:"
    f"\n{OUTPUT_PATH}"
)


# ============================================================
# 18. CLOSE INPUTS
# ============================================================

imerg.close()
glofas.close()
dynamic.close()

del imerg
del glofas
del dynamic
del discharge
del precipitation
del discharge_valid
del valid
del spatial_fraction

gc.collect()


# ============================================================
# 19. REOPEN / VERIFY
# ============================================================

print("\n" + "=" * 80)
print("15. REOPEN VERIFICATION")
print("=" * 80)

check = xr.open_zarr(
    OUTPUT_PATH,
    consolidated=True,
)

print(
    check
)

print(
    "\nDimensions:"
)

print(
    dict(check.sizes)
)

print(
    "\nVariables:"
)

for variable in check.data_vars:

    da = check[
        variable
    ]

    print(
        f"  ✅ {variable:30s}"
        f"{da.shape}"
        f" dtype={da.dtype}"
    )


# ============================================================
# 20. FINAL CHECKS
# ============================================================

print("\nFinal checks...")

assert (
    check.sizes["lat"]
    == NLAT
) if "NLAT" in globals() else True

assert (
    check.sizes["lon"]
    == NLON
) if "NLON" in globals() else True


for variable in check.data_vars:

    if check[
        variable
    ].dtype != np.dtype(
        "float32"
    ):

        print(
            f"⚠️ {variable} is "
            f"{check[variable].dtype}"
        )


print(
    "\n✅ Dynamic core V2 verification finished."
)

check.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("🎉 DYNAMIC CORE V2 READY")
print("=" * 80)

print(
    f"Output:"
)

print(
    OUTPUT_PATH
)

print(
    "\nNext step:"
)

print(
    "Rebuild the training indices/normalization "
    "using dynamic_core_v2.zarr."
)