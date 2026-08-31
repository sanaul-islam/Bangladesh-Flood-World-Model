import os
import xarray as xr
import rioxarray
import numpy as np
import earthaccess
import glob
from pathlib import Path

# ---------- CONFIGURATION ----------
PROJECT_ROOT = "/home/sanaul/Projects/Weather-world-model-project"
OUTPUT_DIR = f"{PROJECT_ROOT}/data/static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bangladesh bounding box
LAT_MIN, LAT_MAX = 20.5, 26.5
LON_MIN, LON_MAX = 88.0, 92.5

print("🔑 Logging in to NASA Earthdata...")
earthaccess.login(strategy="interactive", persist=True)

print("🔍 Searching for SRTM tiles covering Bangladesh...")
results = earthaccess.search_data(
    short_name="SRTMGL1",           # SRTM Global 1 arc‑second
    bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    count=50,                       # Get all tiles
    cloud_hosted=False,             # Download to local disk
)

if not results:
    raise RuntimeError("No SRTM tiles found. Check your Earthdata login or the bounding box.")

print(f"✅ Found {len(results)} SRTM tiles.")

# ---------- DOWNLOAD TILES ----------
download_dir = f"{OUTPUT_DIR}/srtm_tiles"
os.makedirs(download_dir, exist_ok=True)

print("⬇️ Downloading tiles...")
earthaccess.download(results, download_dir)

# ---------- MERGE TILES INTO ONE MOSAIC ----------
print("🔄 Merging tiles into a single raster...")
tif_files = glob.glob(f"{download_dir}/*.tif")
if not tif_files:
    raise FileNotFoundError("No .tif files downloaded. Check the download step.")

# Open and merge using rioxarray
merged = rioxarray.open_mfdataset(tif_files, merge='override', mask_and_scale=True)
# Some SRTM files have 'Band1' as the variable; we rename for clarity
if 'Band1' in merged.variables:
    merged = merged.rename({'Band1': 'elevation'})
merged = merged.rio.clip_box(minx=LON_MIN, miny=LAT_MIN, maxx=LON_MAX, maxy=LAT_MAX)

# Save the merged clipped DEM as a GeoTIFF
merged.rio.to_raster(f"{OUTPUT_DIR}/dem_srtm_bangladesh.tif")
print(f"✅ Merged DEM saved to: {OUTPUT_DIR}/dem_srtm_bangladesh.tif")

# ---------- REGRID TO NASA GRID ----------
print("🔄 Regridding to NASA 0.1° grid...")
# Load your existing NASA cube to get the target grid
cube_path = f"{PROJECT_ROOT}/data/features/master_cube_full.zarr"
if not os.path.exists(cube_path):
    # Fallback: open one NASA .nc4 file to get lat/lon
    import glob
    nasa_file = glob.glob(f"{PROJECT_ROOT}/data/raw/nasa/*.nc4")[0]
    ds_nasa = xr.open_dataset(nasa_file, engine='h5netcdf', group='/')
    lat = ds_nasa.lat.values
    lon = ds_nasa.lon.values
else:
    ds_cube = xr.open_zarr(cube_path, consolidated=True)
    lat = ds_cube.lat.values
    lon = ds_cube.lon.values

# Regrid using bilinear interpolation
dem_regrid = merged.interp(lat=lat, lon=lon, method='linear')
dem_regrid = dem_regrid.rename({'lat': 'lat', 'lon': 'lon'})

# Save as NetCDF for easy integration
dem_regrid.to_netcdf(f"{OUTPUT_DIR}/dem_srtm_regridded.nc")
print(f"✅ Regridded DEM saved to: {OUTPUT_DIR}/dem_srtm_regridded.nc")

# Quick check
print(f"   Elevation range: {dem_regrid.min().values:.1f} – {dem_regrid.max().values:.1f} m")
print("🎉 SRTM download and preprocessing complete!")