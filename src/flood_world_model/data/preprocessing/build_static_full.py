from __future__ import annotations

import gc
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray
from scipy.ndimage import distance_transform_edt


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

print("=" * 80)
print("BANGLADESH FLOOD WORLD MODEL")
print("LOW-RAM STATIC FEATURE BUILDER")
print("=" * 80)

print(f"Working directory: {PROJECT_ROOT}")


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

FLOW_ACC_PATH = (
    HYDRO_DIR
    / "hyd_as_acc_15s"
    / "hyd_as_acc_15s.tif"
)

POPULATION_PATH = (
    POPULATION_DIR
    / "ppp_2013_1km_Aggregated.tif"
)

OUTPUT_PATH = Path(
    "data/features/static.zarr"
)


# ============================================================
# DOMAIN
# ============================================================

LAT_MIN = 20.5
LAT_MAX = 26.5

LON_MIN = 88.0
LON_MAX = 92.5


# ============================================================
# GRID
# ============================================================

print("\nLoading canonical model grid...")

if not DYNAMIC_PATH.exists():
    raise FileNotFoundError(
        f"Dynamic cube not found:\n{DYNAMIC_PATH}"
    )

dynamic = xr.open_zarr(
    DYNAMIC_PATH,
    consolidated=True,
)

target_lat = np.asarray(
    dynamic.lat.values,
    dtype=np.float64,
)

target_lon = np.asarray(
    dynamic.lon.values,
    dtype=np.float64,
)

target_lat = np.sort(target_lat)
target_lon = np.sort(target_lon)

NLAT = len(target_lat)
NLON = len(target_lon)

print(
    f"Target grid: {NLAT} × {NLON}"
)

print(
    f"Latitude: {target_lat.min()} → {target_lat.max()}"
)

print(
    f"Longitude: {target_lon.min()} → {target_lon.max()}"
)

dynamic.close()

del dynamic

gc.collect()


# ============================================================
# UTILITIES
# ============================================================

def run_command(
    command: list[str],
    label: str,
) -> None:
    """
    Run a command while keeping stdout/stderr manageable.
    """

    print(f"   Running: {label}")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(
            "\nGDAL command failed:\n"
            + " ".join(command)
        )

        print(
            "\nSTDERR:\n"
            + result.stderr[-5000:]
        )

        raise RuntimeError(
            f"{label} failed."
        )


def validate_raster(
    path: Path,
    name: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} output not found:\n{path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{name} output is empty."
        )


def open_small_raster(
    path: Path,
    name: str,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Open ONLY the already-resampled small raster.
    Never open the original high-resolution raster here.
    """

    validate_raster(
        path,
        name,
    )

    da = rioxarray.open_rasterio(
        path,
        masked=True,
    ).squeeze(drop=True)

    if da.rio.crs is None:
        raise ValueError(
            f"{name}: missing CRS."
        )

    # Rename spatial coordinates.
    x_dim = da.rio.x_dim
    y_dim = da.rio.y_dim

    if x_dim != "lon" or y_dim != "lat":

        da = da.rename(
            {
                x_dim: "lon",
                y_dim: "lat",
            }
        )

    da = da.sortby("lon")
    da = da.sortby("lat")

    # Ensure exact target grid.
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

    da = da.transpose(
        "lat",
        "lon",
    )

    return da


def gdal_small_raster(
    input_path: Path,
    output_path: Path,
    resampling: str,
    label: str,
) -> None:
    """
    Use GDAL to directly produce a tiny 60×45 raster.

    This avoids loading the source raster into Python.
    """

    run_command(
        [
            "gdalwarp",

            # Input CRS will be respected.
            "-t_srs",
            "EPSG:4326",

            # Exact geographical extent.
            "-te",
            str(LON_MIN),
            str(LAT_MIN),
            str(LON_MAX),
            str(LAT_MAX),

            # EXACT final raster size.
            "-ts",
            str(NLON),
            str(NLAT),

            # Resampling algorithm.
            "-r",
            resampling,

            # Float32 output.
            "-ot",
            "Float32",

            # Don't create unnecessary huge intermediate files.
            "-multi",
            "-wo",
            "NUM_THREADS=1",

            # Compression.
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "PREDICTOR=2",

            str(input_path),
            str(output_path),
        ],
        label,
    )


# ============================================================
# STATIC FEATURE CONTAINER
# ============================================================

layers: dict[str, xr.DataArray] = {}


# ============================================================
# 1. SRTM
# ============================================================

print("\n" + "=" * 80)
print("1. SRTM ELEVATION")
print("=" * 80)

srtm_zip_files = sorted(
    SRTM_DIR.glob("*.hgt.zip")
)

if not srtm_zip_files:
    raise FileNotFoundError(
        f"No SRTM zip files found in {SRTM_DIR}"
    )

print(
    f"Found {len(srtm_zip_files)} SRTM zip files."
)

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    hgt_files: list[str] = []

    # --------------------------------------------------------
    # Extract only temporarily.
    # --------------------------------------------------------

    for zip_file in srtm_zip_files:

        try:

            with zipfile.ZipFile(
                zip_file,
                "r",
            ) as z:

                names = [
                    n
                    for n in z.namelist()
                    if n.lower().endswith(".hgt")
                ]

                if not names:
                    continue

                name = names[0]

                z.extract(
                    name,
                    tmpdir,
                )

                hgt_files.append(
                    str(tmpdir / name)
                )

        except zipfile.BadZipFile:
            print(
                f"⚠️ Bad SRTM zip: {zip_file}"
            )

    if not hgt_files:
        raise RuntimeError(
            "No valid HGT files extracted."
        )

    print(
        f"Extracted {len(hgt_files)} HGT files."
    )

    # --------------------------------------------------------
    # VRT = virtual mosaic.
    #
    # VRT itself is tiny. We do NOT create a huge merged
    # in-memory raster.
    # --------------------------------------------------------

    vrt_path = (
        tmpdir / "srtm_merged.vrt"
    )

    run_command(
        [
            "gdalbuildvrt",
            str(vrt_path),
            *hgt_files,
        ],
        "SRTM VRT mosaic",
    )

    # --------------------------------------------------------
    # Directly create 60 × 45 output.
    # --------------------------------------------------------

    small_dem = (
        tmpdir / "elevation_60x45.tif"
    )

    gdal_small_raster(
        vrt_path,
        small_dem,
        "average",
        "SRTM → 60×45",
    )

    elevation = open_small_raster(
        small_dem,
        "elevation",
        method="linear",
    )

    layers["elevation"] = elevation

    print(
        "   Elevation ready:",
        elevation.shape,
    )

    del elevation

    gc.collect()


# ============================================================
# 2. SLOPE
# ============================================================

print("\n" + "=" * 80)
print("2. SLOPE")
print("=" * 80)

elevation = layers["elevation"]

elev = elevation.values.astype(
    np.float64
)

valid = np.isfinite(elev)

if not valid.any():
    raise RuntimeError(
        "Elevation contains no valid values."
    )

# ------------------------------------------------------------
# Fill small NaN regions temporarily.
# ------------------------------------------------------------

if not valid.all():

    print(
        "   Filling NaNs temporarily for slope..."
    )

    nearest_indices = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )

    filled = elev[
        tuple(nearest_indices)
    ]

else:

    filled = elev


# ------------------------------------------------------------
# Physical grid spacing
# ------------------------------------------------------------

if NLAT < 2 or NLON < 2:
    raise RuntimeError(
        "Target grid is too small."
    )

lat_step = abs(
    float(
        target_lat[1]
        - target_lat[0]
    )
)

lon_step = abs(
    float(
        target_lon[1]
        - target_lon[0]
    )
)

EARTH_M_PER_DEG = 111_320.0

dy_m = (
    lat_step
    * EARTH_M_PER_DEG
)

lat_radians = np.deg2rad(
    target_lat
)

dx_m = (
    lon_step
    * EARTH_M_PER_DEG
    * np.cos(lat_radians)
)


# ------------------------------------------------------------
# Elevation gradients
# ------------------------------------------------------------

dz_dlat = np.gradient(
    filled,
    axis=0,
)

dz_dlon = np.gradient(
    filled,
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
    dz_dx**2
    + dz_dy**2
)

slope_degrees = np.degrees(
    np.arctan(
        slope_gradient
    )
)

slope_degrees[
    ~valid
] = np.nan

slope = xr.DataArray(
    slope_degrees.astype(
        "float32"
    ),
    dims=("lat", "lon"),
    coords={
        "lat": target_lat,
        "lon": target_lon,
    },
    name="slope_degrees",
)

layers["slope_degrees"] = slope

print(
    "   Slope ready:",
    slope.shape,
)

del elev
del filled
del slope_degrees

gc.collect()


# ============================================================
# 3. FLOW ACCUMULATION
# ============================================================

print("\n" + "=" * 80)
print("3. HYDROSHEDS FLOW ACCUMULATION")
print("=" * 80)

if not FLOW_ACC_PATH.exists():
    raise FileNotFoundError(
        f"Flow accumulation not found:\n"
        f"{FLOW_ACC_PATH}"
    )

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    small_acc = (
        tmpdir
        / "flow_accumulation_60x45.tif"
    )

    # "max" preserves high-flow cells better than nearest
    # when reducing a high-resolution hydrological raster
    # to a coarse grid.
    gdal_small_raster(
        FLOW_ACC_PATH,
        small_acc,
        "max",
        "HydroSHEDS → 60×45",
    )

    flow_acc = open_small_raster(
        small_acc,
        "flow_accumulation",
        method="nearest",
    )

    layers[
        "flow_accumulation"
    ] = flow_acc

    print(
        "   Flow accumulation ready:",
        flow_acc.shape,
    )

    del flow_acc

    gc.collect()


# ============================================================
# 4. RIVER MASK
# ============================================================

print("\n" + "=" * 80)
print("4. RIVER MASK")
print("=" * 80)

FLOW_ACC_THRESHOLD = 1000

flow_acc = layers[
    "flow_accumulation"
]

flow_values = flow_acc.values

river_mask_values = (
    np.isfinite(flow_values)
    &
    (
        flow_values
        >= FLOW_ACC_THRESHOLD
    )
)

river_mask = xr.DataArray(
    river_mask_values.astype(
        "float32"
    ),
    dims=("lat", "lon"),
    coords={
        "lat": target_lat,
        "lon": target_lon,
    },
    name="river_mask",
)

layers[
    "river_mask"
] = river_mask

river_cells = int(
    river_mask_values.sum()
)

total_cells = int(
    river_mask_values.size
)

print(
    f"   Threshold: {FLOW_ACC_THRESHOLD}"
)

print(
    f"   River cells: "
    f"{river_cells}/{total_cells}"
)

print(
    f"   Fraction: "
    f"{river_cells / total_cells:.2%}"
)

if river_cells == 0:

    raise RuntimeError(
        "River mask contains ZERO cells. "
        "FLOW_ACC_THRESHOLD is probably too high."
    )


# ============================================================
# 5. RIVER DISTANCE KM
# ============================================================

print("\n" + "=" * 80)
print("5. RIVER DISTANCE")
print("=" * 80)

# ------------------------------------------------------------
# Convert grid distance to approximate km.
#
# distance_transform_edt works on a regular pixel grid.
# We approximate the physical spacing using the mean
# Bangladesh latitude.
# ------------------------------------------------------------

mean_lat = float(
    np.mean(target_lat)
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

distance_pixels = distance_transform_edt(
    ~river_mask_values
)

# We use an average local cell size.
mean_cell_km = np.sqrt(
    dx_km**2
    + dy_km**2
)

river_distance_km = (
    distance_pixels
    * mean_cell_km
)

river_distance_km[
    river_mask_values
] = 0.0

river_distance = xr.DataArray(
    river_distance_km.astype(
        "float32"
    ),
    dims=("lat", "lon"),
    coords={
        "lat": target_lat,
        "lon": target_lon,
    },
    name="river_distance_km",
)

layers[
    "river_distance_km"
] = river_distance

print(
    f"   Mean cell diagonal: "
    f"{mean_cell_km:.2f} km"
)

print(
    "   River distance ready."
)


del distance_pixels
del river_distance_km

gc.collect()


# ============================================================
# 6. LAND COVER
# ============================================================

print("\n" + "=" * 80)
print("6. ESA WORLDCOVER")
print("=" * 80)

landcover_files = sorted(
    LANDCOVER_DIR.glob("*.tif")
)

if not landcover_files:
    raise FileNotFoundError(
        f"No landcover files in {LANDCOVER_DIR}"
    )

print(
    f"Found {len(landcover_files)} land-cover tiles."
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
                for x in landcover_files
            ],
        ],
        "WorldCover VRT mosaic",
    )

    small_lc = (
        tmpdir
        / "landcover_60x45.tif"
    )

    # WorldCover is categorical.
    #
    # DO NOT use average.
    # Nearest preserves the class label.
    gdal_small_raster(
        vrt,
        small_lc,
        "near",
        "WorldCover → 60×45",
    )

    landcover = open_small_raster(
        small_lc,
        "landcover",
        method="nearest",
    )

    layers[
        "landcover"
    ] = landcover

    print(
        "   Landcover ready:",
        landcover.shape,
    )

    del landcover

    gc.collect()


# ============================================================
# 7. SOIL
# ============================================================

print("\n" + "=" * 80)
print("7. SOIL")
print("=" * 80)

soil_files = {
    "soil_clay":
        SOIL_DIR / "T_CLAY.tif",

    "soil_silt":
        SOIL_DIR / "T_SILT.tif",

    "soil_sand":
        SOIL_DIR / "T_SAND.tif",

    "soil_organic_carbon":
        SOIL_DIR / "T_OC.tif",
}

for variable, source_path in soil_files.items():

    print(
        f"\n   {variable}"
    )

    if not source_path.exists():

        raise FileNotFoundError(
            f"Missing soil raster:\n"
            f"{source_path}"
        )

    with tempfile.TemporaryDirectory() as tmp:

        tmpdir = Path(tmp)

        output = (
            tmpdir
            / f"{variable}_60x45.tif"
        )

        gdal_small_raster(
            source_path,
            output,
            "average",
            f"{variable} → 60×45",
        )

        soil = open_small_raster(
            output,
            variable,
            method="linear",
        )

        layers[
            variable
        ] = soil

        print(
            f"   ✅ {variable}: "
            f"{soil.shape}"
        )

        del soil

    gc.collect()


# ============================================================
# 8. POPULATION
# ============================================================

print("\n" + "=" * 80)
print("8. POPULATION")
print("=" * 80)

if not POPULATION_PATH.exists():

    raise FileNotFoundError(
        f"Population raster missing:\n"
        f"{POPULATION_PATH}"
    )

with tempfile.TemporaryDirectory() as tmp:

    tmpdir = Path(tmp)

    small_population = (
        tmpdir
        / "population_60x45.tif"
    )

    # Population is continuous.
    gdal_small_raster(
        POPULATION_PATH,
        small_population,
        "average",
        "Population → 60×45",
    )

    population = open_small_raster(
        small_population,
        "population_density",
        method="linear",
    )

    layers[
        "population_density"
    ] = population

    print(
        "   Population ready:",
        population.shape,
    )

    del population

    gc.collect()


# ============================================================
# 9. FINAL CLEANUP
# ============================================================

print("\n" + "=" * 80)
print("9. FINAL VALIDATION")
print("=" * 80)

# Exact required physical variables.
required_physical = [
    "elevation",
    "slope_degrees",
    "flow_accumulation",
    "river_mask",
    "river_distance_km",
    "landcover",
    "soil_clay",
    "soil_silt",
    "soil_sand",
    "soil_organic_carbon",
]

required_exposure = [
    "population_density",
]

required_all = (
    required_physical
    + required_exposure
)

for variable in required_all:

    if variable not in layers:

        raise RuntimeError(
            f"Required variable missing: "
            f"{variable}"
        )

    da = layers[
        variable
    ]

    # Ensure exact dimensions.
    if da.dims != (
        "lat",
        "lon",
    ):

        raise ValueError(
            f"{variable}: wrong dimensions "
            f"{da.dims}"
        )

    if da.shape != (
        NLAT,
        NLON,
    ):

        raise ValueError(
            f"{variable}: wrong shape "
            f"{da.shape}; expected "
            f"({NLAT},{NLON})"
        )

    # Exact coordinates.
    if not np.allclose(
        da.lat.values,
        target_lat,
    ):

        raise ValueError(
            f"{variable}: latitude mismatch."
        )

    if not np.allclose(
        da.lon.values,
        target_lon,
    ):

        raise ValueError(
            f"{variable}: longitude mismatch."
        )

    layers[
        variable
    ] = da.astype(
        "float32"
    )

    values = da.values

    finite = np.isfinite(
        values
    )

    if finite.any():

        valid_values = values[
            finite
        ]

        print(
            f"✅ {variable:25s}"
            f"min={float(valid_values.min()):10.3f} "
            f"max={float(valid_values.max()):10.3f} "
            f"NaN={int((~finite).sum()):5d}"
        )

    else:

        raise RuntimeError(
            f"{variable}: no finite values."
        )


# ============================================================
# 10. CREATE DATASET
# ============================================================

print("\n" + "=" * 80)
print("10. BUILDING STATIC DATASET")
print("=" * 80)

static = xr.Dataset(
    data_vars=layers,
    coords={
        "lat": target_lat,
        "lon": target_lon,
    },
)

static.attrs = {
    "title": (
        "Bangladesh Flood World Model "
        "Static Features"
    ),
    "grid": "0.1 degree",
    "grid_source": "NASA IMERG grid",
    "crs": "EPSG:4326",
    "domain": (
        f"{LAT_MIN}-{LAT_MAX}N, "
        f"{LON_MIN}-{LON_MAX}E"
    ),
}


# ============================================================
# 11. VARIABLE METADATA
# ============================================================

metadata = {

    "elevation": {
        "role": "physical_static",
        "units": "m",
    },

    "slope_degrees": {
        "role": "physical_static",
        "units": "degrees",
    },

    "flow_accumulation": {
        "role": "physical_static",
    },

    "river_mask": {
        "role": "physical_static",
        "units": "0/1",
    },

    "river_distance_km": {
        "role": "physical_static",
        "units": "km",
    },

    "landcover": {
        "role": "physical_static",
        "description": (
            "ESA WorldCover categorical class."
        ),
    },

    "soil_clay": {
        "role": "physical_static",
    },

    "soil_silt": {
        "role": "physical_static",
    },

    "soil_sand": {
        "role": "physical_static",
    },

    "soil_organic_carbon": {
        "role": "physical_static",
    },

    "population_density": {
        "role": "exposure",
    },
}


for variable, attrs in metadata.items():

    static[
        variable
    ].attrs.update(
        attrs
    )


# ============================================================
# 12. ZARR CHUNKS
# ============================================================

# Static data is tiny (60×45).
#
# There is no point making many chunks.

static = static.chunk(
    {
        "lat": NLAT,
        "lon": NLON,
    }
)


# ============================================================
# 13. REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_PATH.exists():

    print(
        f"\nRemoving old:\n"
        f"{OUTPUT_PATH}"
    )

    shutil.rmtree(
        OUTPUT_PATH
    )


# ============================================================
# 14. SAVE
# ============================================================

print("\n" + "=" * 80)
print("11. SAVING STATIC.ZARR")
print("=" * 80)

static.to_zarr(
    OUTPUT_PATH,
    mode="w",
    consolidated=True,
)

print(
    f"✅ Saved: {OUTPUT_PATH}"
)

static.close()

del static
del layers

gc.collect()


# ============================================================
# 15. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("12. FINAL VERIFICATION")
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
    "\nVariable shapes:"
)

for variable in check.data_vars:

    print(
        f"  ✅ {variable:25s}"
        f"{check[variable].shape}"
        f" dtype={check[variable].dtype}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("🎉 STATIC ZARR COMPLETE")
print("=" * 80)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    f"Grid: {NLAT} × {NLON}"
)

print(
    f"Total variables: "
    f"{len(check.data_vars)}"
)

check.close()

print("\nDone.")