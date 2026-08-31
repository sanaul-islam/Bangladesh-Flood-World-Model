
from __future__ import annotations

import gc
import shutil

import numpy as np
import xarray as xr

from flood_world_model.utils.paths import FEATURES_DIR, PROCESSED_DIR, PROJECT_ROOT


print("=" * 80)
print("BANGLADESH FLOOD WORLD MODEL")
print("LOW-RAM DYNAMIC FEATURE CUBE BUILDER")
print("=" * 80)

print(f"Working directory: {PROJECT_ROOT}")


# ============================================================
# INPUTS
# ============================================================

IMERG_PATH = PROCESSED_DIR / "nasa_imerg_compact.zarr"
GLOFAS_PATH = PROCESSED_DIR / "glofas" / "glofas_2015_2026.zarr"
OUTPUT_PATH = FEATURES_DIR / "dynamic_core.zarr"


# ============================================================
# MODEL GRID
# ============================================================
# IMERG is the canonical grid.

print("\nLoading canonical IMERG grid...")

if not IMERG_PATH.exists():
    raise FileNotFoundError(
        f"IMERG Zarr not found:\n{IMERG_PATH}"
    )

if not GLOFAS_PATH.exists():
    raise FileNotFoundError(
        f"GloFAS Zarr not found:\n{GLOFAS_PATH}"
    )


imerg = xr.open_zarr(
    IMERG_PATH,
    consolidated=True,
)


# ------------------------------------------------------------
# Validate IMERG grid
# ------------------------------------------------------------

if "lat" not in imerg.coords:
    raise ValueError(
        "IMERG does not contain 'lat'."
    )

if "lon" not in imerg.coords:
    raise ValueError(
        "IMERG does not contain 'lon'."
    )

if "time" not in imerg.coords:
    raise ValueError(
        "IMERG does not contain 'time'."
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

NLAT = len(target_lat)
NLON = len(target_lon)

print(
    f"Target grid: "
    f"{NLAT} × {NLON}"
)

print(
    f"Latitude: "
    f"{target_lat.min()} → {target_lat.max()}"
)

print(
    f"Longitude: "
    f"{target_lon.min()} → {target_lon.max()}"
)

print(
    f"IMERG time: "
    f"{imerg.time.min().values} → "
    f"{imerg.time.max().values}"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_coordinates(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Normalize coordinate names to:
        time
        lat
        lon
    """

    rename_map = {}

    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"

    if "latitude" in ds.dims:
        rename_map["latitude"] = "lat"

    if "longitude" in ds.dims:
        rename_map["longitude"] = "lon"

    if "y" in ds.dims and "lat" not in ds.dims:
        rename_map["y"] = "lat"

    if "x" in ds.dims and "lon" not in ds.dims:
        rename_map["x"] = "lon"

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def clean_numeric(
    da: xr.DataArray,
) -> xr.DataArray:
    """
    Convert to float32 and remove +/- infinity.
    """

    da = da.astype("float32")

    da = da.where(
        np.isfinite(da)
    )

    return da


def find_variable(
    ds: xr.Dataset,
    candidates: list[str],
) -> str | None:

    for name in candidates:
        if name in ds.data_vars:
            return name

    return None


def print_stats(
    name: str,
    da: xr.DataArray,
) -> None:

    print(f"\n{name}")

    print(
        f"  dims: {da.dims}"
    )

    print(
        f"  shape: {da.shape}"
    )

    try:
        print(
            f"  dtype: {da.dtype}"
        )

        values = da.values

        valid = np.isfinite(
            values
        )

        if valid.any():

            x = values[
                valid
            ]

            print(
                f"  min: {float(x.min()):.6f}"
            )

            print(
                f"  max: {float(x.max()):.6f}"
            )

            print(
                f"  mean: {float(x.mean()):.6f}"
            )

            print(
                f"  NaN: "
                f"{int((~valid).sum())}"
            )

    except Exception as exc:

        print(
            f"  Statistics skipped: {exc}"
        )


# ============================================================
# 1. PREPARE IMERG
# ============================================================

print("\n" + "=" * 80)
print("1. PREPARING NASA IMERG")
print("=" * 80)

imerg = normalize_coordinates(
    imerg
)

imerg = imerg.sortby(
    "time"
)

imerg = imerg.sortby(
    "lat"
)

imerg = imerg.sortby(
    "lon"
)

if "precipitation" not in imerg.data_vars:

    raise ValueError(
        "IMERG variable 'precipitation' not found.\n"
        f"Available: {list(imerg.data_vars)}"
    )


# ------------------------------------------------------------
# Keep only precipitation
# ------------------------------------------------------------

imerg_core = imerg[
    ["precipitation"]
]


# ------------------------------------------------------------
# Keep only valid model domain.
#
# Usually already cropped, but this makes the code robust.
# ------------------------------------------------------------

imerg_core = imerg_core.sel(
    lat=slice(
        float(target_lat.min()),
        float(target_lat.max()),
    ),
    lon=slice(
        float(target_lon.min()),
        float(target_lon.max()),
    ),
)


# ------------------------------------------------------------
# Convert to float32 lazily.
# ------------------------------------------------------------

imerg_core["precipitation"] = (
    imerg_core["precipitation"]
    .astype("float32")
)


print_stats(
    "IMERG precipitation",
    imerg_core["precipitation"],
)


# ============================================================
# 2. OPEN GLOFAS LAZILY
# ============================================================

print("\n" + "=" * 80)
print("2. PREPARING GloFAS")
print("=" * 80)

glofas = xr.open_zarr(
    GLOFAS_PATH,
    consolidated=True,
)

glofas = normalize_coordinates(
    glofas
)

glofas = glofas.sortby(
    "time"
)

glofas = glofas.sortby(
    "lat"
)

glofas = glofas.sortby(
    "lon"
)

print(
    "Available GloFAS variables:"
)

for name in glofas.data_vars:

    print(
        f"   - {name}"
    )


# ============================================================
# 3. IDENTIFY VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("3. IDENTIFYING HYDROLOGICAL VARIABLES")
print("=" * 80)


aliases = {

    "river_discharge": [
        "river_discharge",
        "discharge",
        "dis24",
        "streamflow",
        "river_dis",
    ],

    "runoff": [
        "runoff",
        "surface_runoff",
        "ro",
        "runoff_mean",
    ],

    "soil_wetness": [
        "soil_wetness",
        "soil_wetness_index",
        "soil_moisture",
        "soilwater",
        "swvl1",
    ],
}


selected = {}

for desired, candidates in aliases.items():

    found = find_variable(
        glofas,
        candidates,
    )

    if found:

        selected[desired] = found

        print(
            f"✅ {desired} ← {found}"
        )

    else:

        print(
            f"⚠️ {desired} not found."
        )


if "river_discharge" not in selected:

    raise RuntimeError(
        "\nGloFAS river discharge could not be identified.\n"
        "Check the variables printed above and modify "
        "the aliases dictionary."
    )


# ============================================================
# 4. SELECT ONLY REQUIRED GLOFAS VARIABLES
# ============================================================

selected_names = list(
    selected.values()
)

glofas_core = glofas[
    selected_names
].copy()


# Rename to stable project names.

glofas_core = glofas_core.rename(
    {
        source: target
        for target, source in selected.items()
    }
)


# ============================================================
# 5. REGRID GLOFAS
# ============================================================

print("\n" + "=" * 80)
print("4. REGRIDDING GloFAS → IMERG GRID")
print("=" * 80)

print(
    "This remains lazy until written to Zarr."
)

glofas_regrid = glofas_core.interp(
    lat=target_lat,
    lon=target_lon,
    method="linear",
)

print(
    f"Regridded shape:"
    f" {dict(glofas_regrid.sizes)}"
)


# ============================================================
# 6. TEMPORAL ALIGNMENT
# ============================================================

print("\n" + "=" * 80)
print("5. ALIGNING TIME")
print("=" * 80)

imerg_aligned, glofas_aligned = xr.align(
    imerg_core,
    glofas_regrid,
    join="inner",
)

print(
    f"Common time steps: "
    f"{imerg_aligned.sizes['time']}"
)

print(
    f"Common range: "
    f"{imerg_aligned.time.min().values} → "
    f"{imerg_aligned.time.max().values}"
)


if imerg_aligned.sizes["time"] == 0:

    raise RuntimeError(
        "No common timestamps between IMERG and GloFAS."
    )


# ============================================================
# 7. MERGE CORE DYNAMIC VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("6. MERGING CORE DYNAMIC DATA")
print("=" * 80)

dynamic = xr.merge(
    [
        imerg_aligned,
        glofas_aligned,
    ],
    join="exact",
    compat="override",
)


# ============================================================
# 8. CLEAN / CAST VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("7. CLEANING VARIABLES")
print("=" * 80)

for variable in dynamic.data_vars:

    dynamic[variable] = (
        clean_numeric(
            dynamic[variable]
        )
    )


# ============================================================
# 9. PRECIPITATION DERIVED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("8. CREATING RAINFALL FEATURES")
print("=" * 80)

precip = dynamic[
    "precipitation"
]


# ------------------------------------------------------------
# 3-day rainfall
#
# Trailing window:
# day t includes t, t-1, t-2.
#
# NO future information.
# ------------------------------------------------------------

dynamic[
    "precip_3d"
] = (
    precip
    .rolling(
        time=3,
        min_periods=1,
    )
    .sum()
    .astype("float32")
)


# ------------------------------------------------------------
# 7-day rainfall
# ------------------------------------------------------------

dynamic[
    "precip_7d"
] = (
    precip
    .rolling(
        time=7,
        min_periods=1,
    )
    .sum()
    .astype("float32")
)


# ------------------------------------------------------------
# Log rainfall
# ------------------------------------------------------------

dynamic[
    "precip_log1p"
] = (
    np.log1p(
        precip.clip(
            min=0
        )
    )
    .astype("float32")
)


# ============================================================
# 10. MISSINGNESS INDICATOR
# ============================================================

dynamic[
    "precip_missing"
] = (
    precip.isnull()
    .astype("float32")
)


# ============================================================
# 11. SORT FINAL VARIABLE ORDER
# ============================================================

preferred = [
    "precipitation",
    "precip_3d",
    "precip_7d",
    "precip_log1p",
    "precip_missing",
    "river_discharge",
    "runoff",
    "soil_wetness",
]

available = [
    x
    for x in preferred
    if x in dynamic.data_vars
]

remaining = [
    x
    for x in dynamic.data_vars
    if x not in available
]

dynamic = dynamic[
    available + remaining
]


# ============================================================
# 12. FINAL GRID VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("9. VALIDATING GRID")
print("=" * 80)

if dynamic.sizes["lat"] != NLAT:

    raise ValueError(
        "Latitude dimension is incorrect."
    )

if dynamic.sizes["lon"] != NLON:

    raise ValueError(
        "Longitude dimension is incorrect."
    )


if not np.allclose(
    dynamic.lat.values,
    target_lat,
):

    raise ValueError(
        "Latitude coordinates do not match IMERG."
    )


if not np.allclose(
    dynamic.lon.values,
    target_lon,
):

    raise ValueError(
        "Longitude coordinates do not match IMERG."
    )


print(
    "✅ Latitude grid matches IMERG."
)

print(
    "✅ Longitude grid matches IMERG."
)


# ============================================================
# 13. TIME VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("10. VALIDATING TIME")
print("=" * 80)

times = dynamic.time.values

print(
    f"Number of timestamps: {len(times)}"
)

print(
    f"Start: {times[0]}"
)

print(
    f"End: {times[-1]}"
)

if len(times) > 1:

    differences = np.diff(
        times
    )

    one_day = np.timedelta64(
        1,
        "D",
    )

    if np.all(
        differences == one_day
    ):

        print(
            "✅ Time series is continuous daily."
        )

    else:

        unique_diffs = np.unique(
            differences
        )

        print(
            "⚠️ Non-daily gaps detected:"
        )

        print(
            unique_diffs[:20]
        )


# ============================================================
# 14. VALIDATE VARIABLE SHAPES
# ============================================================

print("\n" + "=" * 80)
print("11. VARIABLE VALIDATION")
print("=" * 80)

for variable in dynamic.data_vars:

    da = dynamic[
        variable
    ]

    if da.dims != (
        "time",
        "lat",
        "lon",
    ):

        raise ValueError(
            f"{variable} has unexpected dims: "
            f"{da.dims}"
        )

    expected = (
        len(times),
        NLAT,
        NLON,
    )

    if da.shape != expected:

        raise ValueError(
            f"{variable} shape {da.shape} "
            f"!= {expected}"
        )

    print(
        f"✅ {variable:25s}"
        f"{da.shape}"
        f" dtype={da.dtype}"
    )


# ============================================================
# 15. METADATA
# ============================================================

dynamic.attrs = {
    "title": (
        "Bangladesh Flood World Model Dynamic Core"
    ),
    "description": (
        "Daily precipitation and hydrological "
        "variables aligned to the NASA IMERG grid."
    ),
    "canonical_grid": (
        "NASA IMERG 0.1 degree"
    ),
    "time_frequency": "daily",
    "domain": (
        f"{float(target_lat.min())}-"
        f"{float(target_lat.max())} N, "
        f"{float(target_lon.min())}-"
        f"{float(target_lon.max())} E"
    ),
}


# ============================================================
# 16. VARIABLE METADATA
# ============================================================

if "precipitation" in dynamic:

    dynamic[
        "precipitation"
    ].attrs.update(
        {
            "source": "NASA IMERG",
            "role": "dynamic_forcing",
        }
    )


if "precip_3d" in dynamic:

    dynamic[
        "precip_3d"
    ].attrs.update(
        {
            "role": "derived_dynamic_feature",
            "description": (
                "Trailing 3-day precipitation accumulation."
            ),
        }
    )


if "precip_7d" in dynamic:

    dynamic[
        "precip_7d"
    ].attrs.update(
        {
            "role": "derived_dynamic_feature",
            "description": (
                "Trailing 7-day precipitation accumulation."
            ),
        }
    )


if "precip_log1p" in dynamic:

    dynamic[
        "precip_log1p"
    ].attrs.update(
        {
            "role": "derived_dynamic_feature",
            "description": (
                "log1p-transformed precipitation."
            ),
        }
    )


if "precip_missing" in dynamic:

    dynamic[
        "precip_missing"
    ].attrs.update(
        {
            "role": "data_quality_feature",
            "description": (
                "1 where precipitation is missing, "
                "otherwise 0."
            ),
        }
    )


if "river_discharge" in dynamic:

    dynamic[
        "river_discharge"
    ].attrs.update(
        {
            "source": "GloFAS",
            "role": "hydrological_state",
        }
    )


if "runoff" in dynamic:

    dynamic[
        "runoff"
    ].attrs.update(
        {
            "source": "GloFAS",
            "role": "hydrological_state",
        }
    )


if "soil_wetness" in dynamic:

    dynamic[
        "soil_wetness"
    ].attrs.update(
        {
            "source": "GloFAS",
            "role": "hydrological_state",
        }
    )


# ============================================================
# 17. CHUNKING
# ============================================================

print("\n" + "=" * 80)
print("12. APPLYING LAPTOP-FRIENDLY CHUNKS")
print("=" * 80)

# IMPORTANT:
#
# Your training windows are likely 7–30 days.
# So time chunks around 32 are useful.
#
# Spatial dimensions are tiny.

CHUNKS = {
    "time": 32,
    "lat": NLAT,
    "lon": NLON,
}

print(
    "Chunks:",
    CHUNKS,
)

dynamic = dynamic.chunk(
    CHUNKS
)


# ============================================================
# 18. REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_PATH.exists():

    print(
        f"\nRemoving existing output:\n"
        f"{OUTPUT_PATH}"
    )

    shutil.rmtree(
        OUTPUT_PATH
    )


# ============================================================
# 19. SAVE TO ZARR
# ============================================================

print("\n" + "=" * 80)
print("13. WRITING DYNAMIC CORE")
print("=" * 80)

print(
    "Writing lazily/chunked to Zarr..."
)

dynamic.to_zarr(
    OUTPUT_PATH,
    mode="w",
    consolidated=True,
)

print(
    f"✅ Saved:\n{OUTPUT_PATH}"
)


# ============================================================
# 20. FREE INPUT REFERENCES
# ============================================================

imerg.close()
glofas.close()

dynamic.close()

del imerg
del glofas
del imerg_aligned
del glofas_aligned
del dynamic

gc.collect()


# ============================================================
# 21. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("14. FINAL VERIFICATION")
print("=" * 80)

check = xr.open_zarr(
    OUTPUT_PATH,
    consolidated=True,
)

print(
    "Dimensions:",
    dict(check.sizes),
)

print(
    "Variables:",
    list(check.data_vars),
)

print(
    "Time:",
    check.time.min().values,
    "→",
    check.time.max().values,
)


# ------------------------------------------------------------
# Check shapes
# ------------------------------------------------------------

for variable in check.data_vars:

    if check[
        variable
    ].dims != (
        "time",
        "lat",
        "lon",
    ):

        raise ValueError(
            f"Bad dimensions: "
            f"{variable}"
        )


# ------------------------------------------------------------
# Check dtype
# ------------------------------------------------------------

for variable in check.data_vars:

    if check[
        variable
    ].dtype != np.dtype(
        "float32"
    ):

        print(
            f"⚠️ {variable} dtype: "
            f"{check[variable].dtype}"
        )


# ------------------------------------------------------------
# Check grid
# ------------------------------------------------------------

if not np.allclose(
    check.lat.values,
    target_lat,
):

    raise ValueError(
        "Final latitude grid mismatch."
    )

if not np.allclose(
    check.lon.values,
    target_lon,
):

    raise ValueError(
        "Final longitude grid mismatch."
    )


print(
    "✅ Grid verified."
)


# ------------------------------------------------------------
# Approximate disk size
# ------------------------------------------------------------

total_bytes = 0

for path in OUTPUT_PATH.rglob("*"):

    if path.is_file():

        total_bytes += path.stat().st_size


size_mb = (
    total_bytes
    / (1024 ** 2)
)

print(
    f"Final Zarr size: "
    f"{size_mb:.2f} MB"
)


# ------------------------------------------------------------
# Final success
# ------------------------------------------------------------

check.close()

print("\n" + "=" * 80)
print("🎉 DYNAMIC CORE READY")
print("=" * 80)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    f"Grid: {NLAT} × {NLON}"
)

print(
    f"Variables: "
    f"{len(check.data_vars) if False else 'verified'}"
)

print(
    "\nRecommended next step:"
)

print(
    "Build a PyTorch Dataset that reads "
    "14-day windows from this Zarr."
)
