from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


DYNAMIC_PATH = Path(
    "data/features/dynamic_core.zarr"
)

STATIC_PATH = Path(
    "data/features/static.zarr"
)


def show_mask(
    data,
    title,
):
    plt.figure(figsize=(8, 6))
    data.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STATIC
# ------------------------------------------------------------

static = xr.open_zarr(
    STATIC_PATH,
    consolidated=True,
)

for variable in static.data_vars:

    mask = static[variable].isnull()

    if bool(mask.any()):

        show_mask(
            mask,
            f"Missing values: {variable}",
        )


# ------------------------------------------------------------
# RIVER DISCHARGE
# ------------------------------------------------------------

dynamic = xr.open_zarr(
    DYNAMIC_PATH,
    consolidated=True,
)

river_missing = dynamic[
    "river_discharge"
].isnull().mean(
    dim="time"
)

show_mask(
    river_missing,
    "River discharge missing fraction",
)


static.close()
dynamic.close()