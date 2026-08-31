import os
import sys
from pathlib import Path
import xarray as xr
import glob
import numpy as np

# Fix working directory
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
print(f"📂 Working directory: {os.getcwd()}")

# Find all GloFAS NetCDF files
glofas_files = sorted(glob.glob("data/raw/glofas/glofas_*.nc"))
print(f"Found {len(glofas_files)} GloFAS files")

if not glofas_files:
    raise FileNotFoundError("No GloFAS files found")

# Open and combine all files along valid_time
print("Opening files...")
ds = xr.open_mfdataset(
    glofas_files,
    combine='nested',
    concat_dim='valid_time'
)

# Sort by time
ds = ds.sortby('valid_time')

# Rename dimensions to match IMERG grid
ds = ds.rename({
    'valid_time': 'time',
    'latitude': 'lat',
    'longitude': 'lon'
})

# Remove duplicate times
_, idx = np.unique(ds.time.values, return_index=True)
if len(idx) < len(ds.time):
    print(f"Removing {len(ds.time) - len(idx)} duplicate times...")
    ds = ds.isel(time=np.sort(idx))

# Rename variables to match expected names
rename_vars = {}
if 'avg_dis' in ds.variables:
    rename_vars['avg_dis'] = 'river_discharge'
if 'avg_runoff' in ds.variables:
    rename_vars['avg_runoff'] = 'runoff'
if 'soil_wetness_index' in ds.variables:
    rename_vars['soil_wetness_index'] = 'soil_wetness'
if rename_vars:
    ds = ds.rename(rename_vars)

# Rechunk to uniform chunks (Zarr requires uniform chunks except last)
print("Rechunking...")
ds = ds.chunk({'time': 365, 'lat': 120, 'lon': 90})

# Convert to float32
ds = ds.astype('float32')

# Save to Zarr
output_path = "data/interim/glofas/glofas_2015_2026.zarr"
print(f"Saving to {output_path}...")
ds.to_zarr(output_path, mode='w', consolidated=True)
print("✅ GloFAS Zarr saved successfully!")
print(f"   Time range: {ds.time.min().values} → {ds.time.max().values}")
print(f"   Shape: {ds.dims}")
print(f"   Variables: {list(ds.data_vars)}")