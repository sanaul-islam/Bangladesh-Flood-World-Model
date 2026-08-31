import numpy as np
import xarray as xr

LAT_MIN, LAT_MAX = 20.5, 26.5
LON_MIN, LON_MAX = 88.0, 92.5
STEP = 0.1

lat = np.arange(LAT_MIN, LAT_MAX + STEP/2, STEP)
lon = np.arange(LON_MIN, LON_MAX + STEP/2, STEP)

print(f"Grid: {len(lat)} lat × {len(lon)} lon = {len(lat)*len(lon)} cells")

# Save grid for later use
np.save("data/grid/lat.npy", lat)
np.save("data/grid/lon.npy", lon)