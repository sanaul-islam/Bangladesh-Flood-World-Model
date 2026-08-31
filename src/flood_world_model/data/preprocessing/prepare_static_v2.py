from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from scipy.ndimage import distance_transform_edt


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

print("=" * 80)
print("BANGLADESH FLOOD WORLD MODEL")
print("STATIC FEATURE PREPARATION V3 - LOW RAM")
print("=" * 80)

print(f"Project root: {PROJECT_ROOT}")


# ============================================================
# INPUT PATHS
# ============================================================

DYNAMIC_PATH = Path(
    "data/features/dynamic_core.zarr"
)

SRTM_DIR = Path(
    "data/static/srtm_tiles"
)

HYDRO_DIR = Path(
    "data/static/hydrosheds"
)

LANDCOVER_DIR = Path(
    "data/static/landcover"
)

SOIL_DIR = Path(
    "data/static/soil"
)

POPULATION_DIR = Path(
    "data/static/population"
)


# ============================================================
# IMPORTANT PATHS
# ============================================================

FLOW_ACC_PATH = (
    HYDRO_DIR
    / "hyd_as_acc_15s"
    / "hyd_as_acc_15s.tif"
)

POPULATION_PATH = (
    POPULATION_DIR
    / "ppp_2013_1km_Aggregated.tif"
)

# Optional GloFAS structural discharge mask.
GLOFAS_MASK_PATH = Path(
    "data/interim/glofas/"
    "glofas_discharge_valid_mask.zarr"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PATH = Path(
    "data/features/static_v3.zarr"
)

MASK_OUTPUT_PATH = Path(
    "data/features/static_masks_v3.zarr"
)

REPORT_PATH = Path(
    "data/features/analysis/"
    "static_v3_quality_report.json"
)


# ============================================================
# DOMAIN
# ============================================================

LAT_MIN = 20.5
LAT_MAX = 26.5

LON_MIN = 88.0
LON_MAX = 92.5


# ============================================================
# HYDROSHEDS RIVER THRESHOLD
# ============================================================

# This is only a V1/V2 proxy.
# Validate against LGED/river network visually.
FLOW_ACC_THRESHOLD = 1000.0


# ============================================================
# CHECK INPUTS
# ============================================================

required_paths = [
    DYNAMIC_PATH,
    SRTM_DIR,
    HYDRO_DIR,
    LANDCOVER_DIR,
    SOIL_DIR,
    POPULATION_DIR,
    FLOW_ACC_PATH,
    POPULATION_PATH,
]

print("\nChecking input paths...")

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input does not exist:\n{path}"
        )

    print(
        f"  ✅ {path}"
    )


# ============================================================
# LOAD TARGET GRID
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING CANONICAL IMERG GRID")
print("=" * 80)

dynamic_grid = xr.open_zarr(
    DYNAMIC_PATH,
    consolidated=True,
)

target_lat = np.asarray(
    dynamic_grid.lat.values,
    dtype=np.float64,
)

target_lon = np.asarray(
    dynamic_grid.lon.values,
    dtype=np.float64,
)

target_lat = np.sort(target_lat)
target_lon = np.sort(target_lon)

NLAT = len(target_lat)
NLON = len(target_lon)

print(
    f"Grid: {NLAT} × {NLON}"
)

print(
    f"Lat: {target_lat.min()} → "
    f"{target_lat.max()}"
)

print(
    f"Lon: {target_lon.min()} → "
    f"{target_lon.max()}"
)

dynamic_grid.close()

del dynamic_grid

gc.collect()


# ============================================================
# HELPERS
# ============================================================

def run_command(
    command: list[str],
    label: str,
) -> None:

    print(
        f"   ▶ {label}"
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(
            "\nCommand:"
        )

        print(
            " ".join(command)
        )

        print(
            "\nSTDERR:"
        )

        print(
            result.stderr[-6000:]
        )

        raise RuntimeError(
            f"{label} failed."
        )


def gdal_to_model_grid(
    source: Path,
    output: Path,
    resampling: str,
    label: str,
) -> None:

    """
    Reduce huge raster directly to 60 × 45.

    Python never loads the original high-resolution raster.
    """

    run_command(
        [
            "gdalwarp",

            "-t_srs",
            "EPSG:4326",

            "-te",
            str(LON_MIN),
            str(LAT_MIN),
            str(LON_MAX),
            str(LAT_MAX),

            "-ts",
            str(NLON),
            str(NLAT),

            "-r",
            resampling,

            "-ot",
            "Float32",

            "-multi",

            # Conservative memory use.
            "-wm",
            "128",

            "-wo",
            "NUM_THREADS=1",

            "-co",
            "COMPRESS=DEFLATE",

            "-co",
            "PREDICTOR=2",

            "-overwrite",

            str(source),
            str(output),
        ],
        label,
    )


def open_small_raster(
    path: Path,
    name: str,
    method: str = "nearest",
) -> xr.DataArray:

    if not path.exists():

        raise FileNotFoundError(
            f"Small raster not found:\n{path}"
        )

    da = rioxarray.open_rasterio(
        path,
        masked=True,
    ).squeeze(drop=True)

    if da.rio.crs is None:

        raise ValueError(
            f"{name}: missing CRS."
        )

    x_dim = da.rio.x_dim
    y_dim = da.rio.y_dim

    rename_map = {}

    if x_dim != "lon":
        rename_map[x_dim] = "lon"

    if y_dim != "lat":
        rename_map[y_dim] = "lat"

    if rename_map:
        da = da.rename(rename_map)

    da = da.sortby("lon")
    da = da.sortby("lat")

    # Final exact grid.
    da = da.interp(
        lon=target_lon,
        lat=target_lat,
        method=method,
    )

    da = da.astype(
        "float32"
    )

    da = da.where(
        np.isfinite(da)
    )

    da = da.assign_coords(
        lat=target_lat,
        lon=target_lon,
    )

    return da.transpose(
        "lat",
        "lon",
    ).rename(name)


def nearest_fill(
    values: np.ndarray,
    source_valid: np.ndarray,
) -> np.ndarray:
    """
    Fill invalid cells from nearest source-valid cell.
    """

    values = values.astype(
        np.float32,
        copy=True,
    )

    if not source_valid.any():

        raise RuntimeError(
            "No valid source cells available."
        )

    nearest_indices = (
        distance_transform_edt(
            ~source_valid,
            return_distances=False,
            return_indices=True,
        )
    )

    filled = values[
        tuple(nearest_indices)
    ]

    return filled.astype(
        np.float32
    )


def make_da(
    values: np.ndarray,
    name: str,
) -> xr.DataArray:

    return xr.DataArray(
        values.astype(
            np.float32
        ),
        dims=("lat", "lon"),
        coords={
            "lat": target_lat,
            "lon": target_lon,
        },
        name=name,
    )


# ============================================================
# CONTAINERS
# ============================================================

layers: dict[str, xr.DataArray] = {}
masks: dict[str, xr.DataArray] = {}


# ============================================================
# 2. ELEVATION
# ============================================================

print("\n" + "=" * 80)
print("2. ELEVATION")
print("=" * 80)

srtm_files = sorted(
    SRTM_DIR.glob("*.hgt.zip")
)

if not srtm_files:

    raise FileNotFoundError(
        f"No *.hgt.zip files in {SRTM_DIR}"
    )

print(
    f"Found {len(srtm_files)} SRTM tiles."
)

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    hgt_files = []

    for zip_path in srtm_files:

        try:

            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as archive:

                names = [
                    x
                    for x in archive.namelist()
                    if x.lower().endswith(".hgt")
                ]

                if not names:
                    continue

                name = names[0]

                archive.extract(
                    name,
                    tmpdir,
                )

                hgt_files.append(
                    str(
                        tmpdir / name
                    )
                )

        except zipfile.BadZipFile:

            print(
                f"⚠️ Bad ZIP: {zip_path}"
            )

    if not hgt_files:

        raise RuntimeError(
            "No HGT files extracted."
        )

    vrt = (
        tmpdir
        / "srtm.vrt"
    )

    run_command(
        [
            "gdalbuildvrt",
            str(vrt),
            *hgt_files,
        ],
        "Build SRTM VRT",
    )

    small_dem = (
        tmpdir
        / "elevation.tif"
    )

    gdal_to_model_grid(
        vrt,
        small_dem,
        "average",
        "SRTM → model grid",
    )

    elevation_da = (
        open_small_raster(
            small_dem,
            "elevation",
            method="linear",
        )
    )


# ------------------------------------------------------------
# Original DEM validity
# ------------------------------------------------------------

elevation_raw = (
    elevation_da
    .values
    .astype(np.float32)
)

elevation_valid = np.isfinite(
    elevation_raw
)

# Tiny numerical negative values are treated as zero.
tiny_negative = (
    (elevation_raw < 0)
    &
    (elevation_raw > -0.01)
)

elevation_raw[
    tiny_negative
] = 0.0


# ------------------------------------------------------------
# LAND MASK
# ------------------------------------------------------------

land_mask = (
    elevation_valid
)

print(
    f"Land cells: "
    f"{land_mask.sum():,}/"
    f"{land_mask.size:,}"
)

print(
    f"Land fraction: "
    f"{land_mask.mean():.2%}"
)


# ------------------------------------------------------------
# Fill elevation ON LAND ONLY.
#
# Non-land remains zero.
# ------------------------------------------------------------

elevation_sources = (
    elevation_valid
    & land_mask
)

if (
    (~elevation_sources & land_mask).any()
):

    elevation_filled = nearest_fill(
        elevation_raw,
        elevation_sources,
    )

else:

    elevation_filled = elevation_raw.copy()


elevation_filled[
    ~land_mask
] = 0.0

layers[
    "elevation"
] = make_da(
    elevation_filled,
    "elevation",
)

masks[
    "elevation_valid"
] = make_da(
    elevation_valid.astype(
        np.float32
    ),
    "elevation_valid",
)

layers[
    "land_mask"
] = make_da(
    land_mask.astype(
        np.float32
    ),
    "land_mask",
)

masks[
    "land_mask"
] = layers[
    "land_mask"
]

print(
    "✅ Elevation ready."
)

del elevation_da
del elevation_raw
del elevation_filled
del elevation_sources

gc.collect()


# ============================================================
# 3. SLOPE
# ============================================================

print("\n" + "=" * 80)
print("3. SLOPE")
print("=" * 80)

elev = layers[
    "elevation"
].values.astype(
    np.float64
)

# ------------------------------------------------------------
# Gradient on final 0.1 degree grid.
# ------------------------------------------------------------

lat_step = abs(
    target_lat[1]
    - target_lat[0]
)

lon_step = abs(
    target_lon[1]
    - target_lon[0]
)

METERS_PER_DEGREE = 111_320.0

dy_m = (
    lat_step
    * METERS_PER_DEGREE
)

dx_m = (
    lon_step
    * METERS_PER_DEGREE
    * np.cos(
        np.deg2rad(
            target_lat
        )
    )
)

dz_dlat = np.gradient(
    elev,
    axis=0,
)

dz_dlon = np.gradient(
    elev,
    axis=1,
)

dz_dy = (
    dz_dlat
    / dy_m
)

dz_dx = (
    dz_dlon
    / dx_m[:, None]
)

slope_gradient = np.sqrt(
    dz_dx ** 2
    + dz_dy ** 2
)

slope_degrees = np.degrees(
    np.arctan(
        slope_gradient
    )
)

# Outside land = 0.
slope_degrees[
    ~land_mask
] = 0.0

# Numerical safety.
slope_degrees = np.nan_to_num(
    slope_degrees,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

slope = make_da(
    slope_degrees,
    "slope_degrees",
)

layers[
    "slope_degrees"
] = slope

masks[
    "slope_valid"
] = make_da(
    land_mask.astype(
        np.float32
    ),
    "slope_valid",
)

print(
    f"Slope min: "
    f"{slope_degrees.min():.4f}"
)

print(
    f"Slope max: "
    f"{slope_degrees.max():.4f}"
)

del elev
del dz_dlat
del dz_dlon
del dz_dy
del dz_dx
del slope_gradient
del slope_degrees

gc.collect()


# ============================================================
# 4. FLOW ACCUMULATION
# ============================================================

print("\n" + "=" * 80)
print("4. FLOW ACCUMULATION")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    small_flow = (
        tmpdir
        / "flow_accumulation.tif"
    )

    # IMPORTANT:
    # average instead of max.
    gdal_to_model_grid(
        FLOW_ACC_PATH,
        small_flow,
        "average",
        "HydroSHEDS → model grid",
    )

    flow_da = open_small_raster(
        small_flow,
        "flow_accumulation",
        method="nearest",
    )


flow_raw = (
    flow_da
    .values
    .astype(np.float32)
)

flow_valid = np.isfinite(
    flow_raw
)

# ------------------------------------------------------------
# Fill missing flow accumulation inside land.
# ------------------------------------------------------------

flow_sources = (
    flow_valid
    & land_mask
)

if (
    (~flow_sources & land_mask).any()
):

    flow_filled = nearest_fill(
        flow_raw,
        flow_sources,
    )

else:

    flow_filled = flow_raw.copy()


# Non-land = zero.
flow_filled[
    ~land_mask
] = 0.0

# No negative flow accumulation.
flow_filled = np.maximum(
    flow_filled,
    0.0,
)

layers[
    "flow_accumulation"
] = make_da(
    flow_filled,
    "flow_accumulation",
)

masks[
    "flow_accumulation_valid"
] = make_da(
    flow_valid.astype(
        np.float32
    ),
    "flow_accumulation_valid",
)

print(
    f"Original valid: "
    f"{flow_valid.sum():,}/"
    f"{flow_valid.size:,}"
)

print(
    f"Final NaN: "
    f"{np.isnan(flow_filled).sum():,}"
)

print(
    f"Min: "
    f"{flow_filled.min():.3f}"
)

print(
    f"Max: "
    f"{flow_filled.max():.3f}"
)

del flow_da
del flow_raw
del flow_filled
del flow_valid
del flow_sources

gc.collect()


# ============================================================
# 5. HYDROSHEDS RIVER MASK
# ============================================================

print("\n" + "=" * 80)
print("5. HYDROSHEDS RIVER MASK")
print("=" * 80)

flow_values = (
    layers[
        "flow_accumulation"
    ].values
)

river_mask_values = (
    land_mask
    &
    np.isfinite(
        flow_values
    )
    &
    (
        flow_values
        >= FLOW_ACC_THRESHOLD
    )
)

river_mask = make_da(
    river_mask_values.astype(
        np.float32
    ),
    "river_mask",
)

river_mask.attrs = {
    "source": "HydroSHEDS",
    "threshold":
        FLOW_ACC_THRESHOLD,
    "role": "physical_static",
}

layers[
    "river_mask"
] = river_mask

river_cells = int(
    river_mask_values.sum()
)

river_fraction = (
    river_cells
    / river_mask_values.size
)

print(
    f"River threshold: "
    f"{FLOW_ACC_THRESHOLD}"
)

print(
    f"River cells: "
    f"{river_cells:,}/"
    f"{river_mask_values.size:,}"
)

print(
    f"River fraction: "
    f"{river_fraction:.2%}"
)


# ============================================================
# 6. GloFAS DISCHARGE DOMAIN MASK
# ============================================================

print("\n" + "=" * 80)
print("6. GloFAS DISCHARGE DOMAIN MASK")
print("=" * 80)

if GLOFAS_MASK_PATH.exists():

    gf = xr.open_zarr(
        GLOFAS_MASK_PATH,
        consolidated=True,
    )

    if (
        "glofas_discharge_valid_mask"
        not in gf
    ):

        print(
            "⚠️ Expected GloFAS mask variable "
            "not found. Skipping."
        )

    else:

        gf_mask = gf[
            "glofas_discharge_valid_mask"
        ]

        gf_mask = (
            gf_mask
            .interp(
                lat=target_lat,
                lon=target_lon,
                method="nearest",
            )
            .compute()
        )

        gf_values = (
            gf_mask
            .values
            .astype(np.float32)
        )

        gf_values = (
            np.isfinite(
                gf_values
            )
            &
            (gf_values >= 0.5)
        ).astype(
            np.float32
        )

        layers[
            "glofas_discharge_domain_mask"
        ] = make_da(
            gf_values,
            "glofas_discharge_domain_mask",
        )

        masks[
            "glofas_discharge_domain_mask"
        ] = layers[
            "glofas_discharge_domain_mask"
        ]

        print(
            f"GloFAS domain cells: "
            f"{int(gf_values.sum()):,}/"
            f"{gf_values.size:,}"
        )

        del gf_mask
        del gf_values

    gf.close()

else:

    print(
        "⚠️ GloFAS mask not found."
    )

gc.collect()


# ============================================================
# 7. RIVER DISTANCE
# ============================================================

print("\n" + "=" * 80)
print("7. RIVER DISTANCE")
print("=" * 80)

distance_pixels = distance_transform_edt(
    ~river_mask_values
)

mean_lat = float(
    target_lat.mean()
)

dy_km = (
    lat_step
    * 111.32
)

dx_km = (
    lon_step
    * 111.32
    * np.cos(
        np.deg2rad(mean_lat)
    )
)

# We use an approximate physical scale.
cell_km = np.sqrt(
    dx_km ** 2
    + dy_km ** 2
)

river_distance_km = (
    distance_pixels
    * cell_km
)

river_distance_km[
    river_mask_values
] = 0.0

river_distance_km = np.nan_to_num(
    river_distance_km,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

layers[
    "river_distance_km"
] = make_da(
    river_distance_km,
    "river_distance_km",
)

print(
    f"Max distance: "
    f"{river_distance_km.max():.2f} km"
)

del distance_pixels
del river_distance_km

gc.collect()


# ============================================================
# 8. LAND COVER
# ============================================================

print("\n" + "=" * 80)
print("8. LAND COVER")
print("=" * 80)

lc_files = sorted(
    LANDCOVER_DIR.glob("*.tif")
)

if not lc_files:

    raise FileNotFoundError(
        f"No landcover TIFF files in "
        f"{LANDCOVER_DIR}"
    )

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    vrt = (
        tmpdir
        / "landcover.vrt"
    )

    run_command(
        [
            "gdalbuildvrt",
            str(vrt),
            *[
                str(x)
                for x in lc_files
            ],
        ],
        "Build WorldCover VRT",
    )

    small_lc = (
        tmpdir
        / "landcover.tif"
    )

    gdal_to_model_grid(
        vrt,
        small_lc,
        "near",
        "WorldCover → model grid",
    )

    lc = open_small_raster(
        small_lc,
        "landcover",
        method="nearest",
    )

lc_values = (
    lc.values
    .astype(np.float32)
)

lc_valid = np.isfinite(
    lc_values
)

lc_sources = (
    lc_valid
    & land_mask
)

if (
    (~lc_sources & land_mask).any()
):

    lc_filled = nearest_fill(
        lc_values,
        lc_sources,
    )

else:

    lc_filled = lc_values.copy()

# Outside land -> 0.
lc_filled[
    ~land_mask
] = 0.0

# Ensure valid finite values.
lc_filled = np.nan_to_num(
    lc_filled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

layers[
    "landcover"
] = make_da(
    lc_filled,
    "landcover",
)

masks[
    "landcover_valid"
] = make_da(
    lc_valid.astype(
        np.float32
    ),
    "landcover_valid",
)

print(
    f"Valid original landcover: "
    f"{lc_valid.sum():,}/"
    f"{lc_valid.size:,}"
)

print(
    "✅ Landcover ready."
)

del lc
del lc_values
del lc_valid
del lc_sources
del lc_filled

gc.collect()


# ============================================================
# 9. SOIL
# ============================================================

print("\n" + "=" * 80)
print("9. SOIL")
print("=" * 80)

soil_sources = {

    "soil_clay":
        SOIL_DIR / "T_CLAY.tif",

    "soil_silt":
        SOIL_DIR / "T_SILT.tif",

    "soil_sand":
        SOIL_DIR / "T_SAND.tif",

    "soil_organic_carbon":
        SOIL_DIR / "T_OC.tif",
}


for variable, source in soil_sources.items():

    print(
        f"\nProcessing {variable}..."
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Missing soil file:\n{source}"
        )

    with tempfile.TemporaryDirectory() as tmp:

        tmpdir = Path(tmp)

        small = (
            tmpdir
            / f"{variable}.tif"
        )

        gdal_to_model_grid(
            source,
            small,
            "average",
            f"{variable} → model grid",
        )

        soil = open_small_raster(
            small,
            variable,
            method="linear",
        )


    values = (
        soil.values
        .astype(np.float32)
    )

    valid = np.isfinite(
        values
    )

    source_valid = (
        valid
        & land_mask
    )

    if (
        (~source_valid & land_mask).any()
    ):

        filled = nearest_fill(
            values,
            source_valid,
        )

    else:

        filled = values.copy()

    filled[
        ~land_mask
    ] = 0.0

    filled = np.nan_to_num(
        filled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    layers[
        variable
    ] = make_da(
        filled,
        variable,
    )

    masks[
        f"{variable}_valid"
    ] = make_da(
        valid.astype(
            np.float32
        ),
        f"{variable}_valid",
    )

    print(
        f"Original valid: "
        f"{valid.sum():,}/"
        f"{valid.size:,}"
    )

    print(
        f"Final NaN: "
        f"{np.isnan(filled).sum():,}"
    )

    del soil
    del values
    del valid
    del source_valid
    del filled

    gc.collect()


# ============================================================
# 10. POPULATION
# ============================================================

print("\n" + "=" * 80)
print("10. POPULATION")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    small_pop = (
        tmpdir
        / "population.tif"
    )

    gdal_to_model_grid(
        POPULATION_PATH,
        small_pop,
        "average",
        "Population → model grid",
    )

    population = open_small_raster(
        small_pop,
        "population_density",
        method="linear",
    )


pop_values = (
    population.values
    .astype(np.float32)
)

pop_valid = np.isfinite(
    pop_values
)

pop_sources = (
    pop_valid
    & land_mask
)

if (
    (~pop_sources & land_mask).any()
):

    pop_filled = nearest_fill(
        pop_values,
        pop_sources,
    )

else:

    pop_filled = pop_values.copy()


# Population cannot be negative.
pop_filled = np.maximum(
    pop_filled,
    0.0,
)

pop_filled[
    ~land_mask
] = 0.0

pop_filled = np.nan_to_num(
    pop_filled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

layers[
    "population_density"
] = make_da(
    pop_filled,
    "population_density",
)

masks[
    "population_valid"
] = make_da(
    pop_valid.astype(
        np.float32
    ),
    "population_valid",
)

print(
    f"Original valid population: "
    f"{pop_valid.sum():,}/"
    f"{pop_valid.size:,}"
)

print(
    "✅ Population ready."
)

del population
del pop_values
del pop_valid
del pop_sources
del pop_filled

gc.collect()


# ============================================================
# 11. FINAL DATASET CONSTRUCTION
# ============================================================

print("\n" + "=" * 80)
print("11. BUILDING FINAL STATIC DATASET")
print("=" * 80)

static = xr.Dataset(
    data_vars=layers,
    coords={
        "lat": target_lat,
        "lon": target_lon,
    },
)

static_masks = xr.Dataset(
    data_vars=masks,
    coords={
        "lat": target_lat,
        "lon": target_lon,
    },
)


# ============================================================
# 12. METADATA
# ============================================================

static.attrs = {

    "title":
        "Bangladesh Flood World Model "
        "Static Features V3",

    "grid":
        "0.1 degree",

    "grid_source":
        "NASA IMERG",

    "crs":
        "EPSG:4326",

    "domain":
        (
            f"{LAT_MIN}-{LAT_MAX}N, "
            f"{LON_MIN}-{LON_MAX}E"
        ),

    "river_mask_source":
        "HydroSHEDS flow accumulation",

    "river_mask_threshold":
        str(FLOW_ACC_THRESHOLD),

    "missing_value_policy":
        (
            "Values missing on valid land cells are "
            "nearest-filled. Non-land cells are set "
            "to zero and represented by land_mask."
        ),
}


# ============================================================
# 13. FINAL FINITE CHECK
# ============================================================

print("\n" + "=" * 80)
print("12. FINAL FINITE CHECK")
print("=" * 80)

report = {
    "grid": {
        "lat": NLAT,
        "lon": NLON,
    },
    "variables": {},
    "land_fraction":
        float(
            land_mask.mean()
        ),
    "river_fraction":
        float(
            river_fraction
        ),
}


for variable in static.data_vars:

    values = static[
        variable
    ].values

    nan_count = int(
        np.isnan(values).sum()
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    finite = np.isfinite(
        values
    )

    print(
        f"{variable:40s}"
        f"NaN={nan_count:4d} "
        f"Inf={inf_count:4d}"
    )

    report[
        "variables"
    ][variable] = {

        "shape":
            list(values.shape),

        "dtype":
            str(values.dtype),

        "nan":
            nan_count,

        "inf":
            inf_count,

        "finite":
            int(finite.sum()),

        "valid_fraction":
            float(finite.mean()),
    }

    if not finite.all():

        raise RuntimeError(
            f"{variable} still contains "
            "NaN or Inf."
        )


# ============================================================
# 14. PHYSICAL CHECKS
# ============================================================

print("\n" + "=" * 80)
print("13. PHYSICAL CHECKS")
print("=" * 80)


# Elevation
elevation_values = (
    static[
        "elevation"
    ].values
)

print(
    f"Elevation: "
    f"{elevation_values.min():.3f} → "
    f"{elevation_values.max():.3f}"
)


# Slope
slope_values = (
    static[
        "slope_degrees"
    ].values
)

print(
    f"Slope: "
    f"{slope_values.min():.3f} → "
    f"{slope_values.max():.3f}"
)

if slope_values.min() < 0:
    raise RuntimeError(
        "Negative slope."
    )

if slope_values.max() > 90:
    raise RuntimeError(
        "Slope > 90 degrees."
    )


# Flow accumulation
flow_values = (
    static[
        "flow_accumulation"
    ].values
)

if flow_values.min() < 0:

    raise RuntimeError(
        "Negative flow accumulation."
    )


# Population
population_values = (
    static[
        "population_density"
    ].values
)

if population_values.min() < 0:

    raise RuntimeError(
        "Negative population."
    )


# River mask
river_values = (
    static[
        "river_mask"
    ].values
)

unique_river = np.unique(
    river_values
)

print(
    "River mask values:",
    unique_river,
)


# Land mask
land_values = (
    static[
        "land_mask"
    ].values
)

print(
    "Land mask values:",
    np.unique(
        land_values
    ),
)


# ============================================================
# 15. REMOVE PREVIOUS OUTPUT
# ============================================================

for path in [
    OUTPUT_PATH,
    MASK_OUTPUT_PATH,
]:

    if path.exists():

        print(
            f"Removing old output: "
            f"{path}"
        )

        shutil.rmtree(
            path
        )


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 16. SAVE
# ============================================================

print("\n" + "=" * 80)
print("14. SAVING STATIC V3")
print("=" * 80)

static = static.chunk(
    {
        "lat": NLAT,
        "lon": NLON,
    }
)

static_masks = static_masks.chunk(
    {
        "lat": NLAT,
        "lon": NLON,
    }
)

static.to_zarr(
    OUTPUT_PATH,
    mode="w",
    consolidated=True,
)

static_masks.to_zarr(
    MASK_OUTPUT_PATH,
    mode="w",
    consolidated=True,
)


with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


print(
    f"✅ Static:"
    f"\n   {OUTPUT_PATH}"
)

print(
    f"✅ Masks:"
    f"\n   {MASK_OUTPUT_PATH}"
)

print(
    f"✅ Report:"
    f"\n   {REPORT_PATH}"
)


# ============================================================
# 17. REOPEN VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("15. REOPEN VERIFICATION")
print("=" * 80)

static.close()
static_masks.close()

del static
del static_masks

gc.collect()


check = xr.open_zarr(
    OUTPUT_PATH,
    consolidated=True,
)

mask_check = xr.open_zarr(
    MASK_OUTPUT_PATH,
    consolidated=True,
)

print(
    "Final static dataset:"
)

print(
    check
)

print(
    "\nFinal mask dataset:"
)

print(
    mask_check
)


# Exact dimensions.
assert (
    check.sizes["lat"]
    == NLAT
)

assert (
    check.sizes["lon"]
    == NLON
)


# Exact coordinates.
assert np.allclose(
    check.lat.values,
    target_lat,
)

assert np.allclose(
    check.lon.values,
    target_lon,
)


# All finite.
for variable in check.data_vars:

    values = check[
        variable
    ].values

    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Verification failed: "
            f"{variable}"
        )


print(
    "\n✅ All static variables are finite."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("🎉 STATIC V3 READY")
print("=" * 80)

print(
    f"Grid: {NLAT} × {NLON}"
)

print(
    f"Land fraction: "
    f"{land_mask.mean():.2%}"
)

print(
    f"River fraction: "
    f"{river_fraction:.2%}"
)

print(
    "\nVariables:"
)

for variable in check.data_vars:

    print(
        f"  ✅ {variable}"
    )

print(
    "\nMasks:"
)

for variable in mask_check.data_vars:

    print(
        f"  ✅ {variable}"
    )

check.close()
mask_check.close()

print(
    "\nDone."
)