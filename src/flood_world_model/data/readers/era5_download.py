
from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path("data/raw/era5_land")
PROCESSED_DIR = Path("data/processed/era5_land")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ZARR = (
    PROCESSED_DIR / "era5_land_daily_2015_2026.zarr"
)

# Bangladesh / relevant surrounding region.
#
# North, West, South, East
#
# This is slightly larger than Bangladesh so you have some
# surrounding context.
AREA = [
    27.5,   # North
    87.0,   # West
    20.0,   # South
    93.5,   # East
]

START_YEAR = 2022
END_YEAR = 2026

# If you only want data through the latest known date,
# set END_YEAR accordingly.
#
# ERA5-Land data availability can lag behind the present.
# The downloader will record failures rather than crashing.

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]

# ============================================================
# Important:
#
# DO NOT download total_precipitation here.
#
# You already use NASA IMERG for precipitation.
#
# DO NOT download soil moisture here if GloFAS soil wetness
# is your chosen hydrological state variable.
# ============================================================


# Final Zarr chunks
ZARR_CHUNKS = {
    "time": 32,
    "lat": 60,
    "lon": 65,
}


# ============================================================
# Helpers
# ============================================================

def days_in_month(year: int, month: int) -> int:
    """Return number of days in month."""
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    current_month = datetime(year, month, 1)

    return (next_month - current_month).days


def temporary_path(year: int, month: int) -> Path:
    return RAW_DIR / f"era5_land_{year}_{month:02d}.nc"


def zarr_path() -> Path:
    return OUTPUT_ZARR


def build_request(year: int, month: int) -> dict:
    num_days = days_in_month(year, month)

    return {
        "variable": VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [
            f"{day:02d}"
            for day in range(1, num_days + 1)
        ],
        "time": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "area": AREA,
        "format": "netcdf",
    }


def is_valid_file(path: Path, minimum_mb: float = 5.0) -> bool:
    if not path.exists():
        return False

    size_mb = path.stat().st_size / (1024 * 1024)

    return size_mb >= minimum_mb


# ============================================================
# Daily processing
# ============================================================

def process_month(
    file_path: Path,
    year: int,
    month: int,
) -> xr.Dataset:
    """
    Open one monthly ERA5-Land file and reduce hourly data
    to daily features.

    Output dimensions:
        time, lat, lon
    """

    print(f"   🔄 Processing {file_path.name}")

    with xr.open_dataset(
        file_path,
        engine="netcdf4",
    ) as ds:

        # ----------------------------------------------------
        # Make dimension ordering consistent
        # ----------------------------------------------------
        ds = ds.transpose(
            "time",
            "lat",
            "lon",
        )

        # ----------------------------------------------------
        # Sort coordinates
        # ----------------------------------------------------
        if "lat" in ds.coords:
            ds = ds.sortby("lat")

        if "lon" in ds.coords:
            ds = ds.sortby("lon")

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------
        for variable in ds.data_vars:
            ds[variable] = ds[variable].astype(
                "float32"
            )

        # ----------------------------------------------------
        # Remove invalid infinite values
        # ----------------------------------------------------
        for variable in ds.data_vars:
            ds[variable] = ds[variable].where(
                np.isfinite(ds[variable])
            )

        # ----------------------------------------------------
        # Daily aggregation
        # ----------------------------------------------------
        daily = xr.Dataset()

        if "t2m" in ds:
            daily["temperature_2m"] = (
                ds["t2m"]
                .resample(time="1D")
                .mean()
                .astype("float32")
            )

        if "u10" in ds:
            daily["wind_u_10m"] = (
                ds["u10"]
                .resample(time="1D")
                .mean()
                .astype("float32")
            )

        if "v10" in ds:
            daily["wind_v_10m"] = (
                ds["v10"]
                .resample(time="1D")
                .mean()
                .astype("float32")
            )

        if "sp" in ds:
            daily["surface_pressure"] = (
                ds["sp"]
                .resample(time="1D")
                .mean()
                .astype("float32")
            )

    # --------------------------------------------------------
    # Derived wind variables
    # --------------------------------------------------------

    if (
        "wind_u_10m" in daily
        and "wind_v_10m" in daily
    ):
        daily["wind_speed_10m"] = np.sqrt(
            daily["wind_u_10m"] ** 2
            + daily["wind_v_10m"] ** 2
        ).astype("float32")

        # Meteorological direction is conventionally expressed
        # as the direction FROM which the wind blows.
        direction = (
            np.degrees(
                np.arctan2(
                    -daily["wind_u_10m"],
                    -daily["wind_v_10m"],
                )
            )
            % 360
        )

        daily["wind_direction_10m"] = (
            direction.astype("float32")
        )

    return daily


# ============================================================
# Append dataset to Zarr
# ============================================================

def append_to_zarr(
    ds: xr.Dataset,
    output_path: Path,
    first_write: bool,
) -> None:

    ds = ds.chunk(ZARR_CHUNKS)

    if first_write:
        print("   💾 Creating ERA5-Land Zarr...")

        ds.to_zarr(
            output_path,
            mode="w",
            consolidated=True,
        )

    else:
        print("   ➕ Appending to ERA5-Land Zarr...")

        ds.to_zarr(
            output_path,
            mode="a",
            append_dim="time",
            consolidated=False,
        )


# ============================================================
# Main downloader
# ============================================================

def main():

    print("=" * 72)
    print("ERA5-Land → Daily Compact Dataset")
    print("=" * 72)

    print(f"Period : {START_YEAR} → {END_YEAR}")
    print(f"Area   : {AREA}")
    print(f"Variables: {VARIABLES}")
    print()

    client = cdsapi.Client()

    first_write = not OUTPUT_ZARR.exists()

    failed_months: list[str] = []
    downloaded_months = 0
    processed_months = 0

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        for month in range(1, 13):

            month_name = f"{year}-{month:02d}"

            print()
            print("-" * 72)
            print(f"📅 {month_name}")
            print("-" * 72)

            target = temporary_path(
                year,
                month,
            )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            if is_valid_file(target):

                print(
                    f"⏭️ Existing file found: "
                    f"{target.name}"
                )

            else:

                request = build_request(
                    year,
                    month,
                )

                success = False

                for attempt in range(1, 6):

                    try:

                        print(
                            f"⬇️ Download attempt "
                            f"{attempt}/5..."
                        )

                        client.retrieve(
                            "reanalysis-era5-land",
                            request,
                            str(target),
                        )

                        if is_valid_file(target):

                            size_mb = (
                                target.stat().st_size
                                / (1024 * 1024)
                            )

                            print(
                                f"✅ Downloaded "
                                f"({size_mb:.1f} MB)"
                            )

                            downloaded_months += 1

                            success = True

                            break

                        print(
                            "⚠️ File is too small. "
                            "Removing..."
                        )

                        if target.exists():
                            target.unlink()

                    except Exception as exc:

                        print(
                            f"❌ Attempt {attempt} failed:"
                            f" {exc}"
                        )

                        if attempt < 5:

                            wait = min(
                                60 * attempt,
                                300,
                            )

                            print(
                                f"   Waiting {wait}s..."
                            )

                            time.sleep(wait)

                if not success:

                    failed_months.append(
                        month_name
                    )

                    print(
                        f"❌ Could not download "
                        f"{month_name}"
                    )

                    continue

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            try:

                daily = process_month(
                    target,
                    year,
                    month,
                )

                print(
                    f"   ✅ Daily shape: "
                    f"{dict(daily.sizes)}"
                )

                print(
                    f"   Variables: "
                    f"{list(daily.data_vars)}"
                )

                # --------------------------------------------
                # Ensure one month is actually loaded and
                # released before the next month.
                # --------------------------------------------

                daily = daily.load()

                # --------------------------------------------
                # Append to compact Zarr
                # --------------------------------------------

                append_to_zarr(
                    daily,
                    OUTPUT_ZARR,
                    first_write,
                )

                first_write = False

                processed_months += 1

                # --------------------------------------------
                # Free memory
                # --------------------------------------------

                daily.close()

                del daily

                # --------------------------------------------
                # Delete temporary raw month
                #
                # This is the major storage-saving step.
                # --------------------------------------------

                try:

                    target.unlink()

                    print(
                        f"   🗑️ Removed temporary "
                        f"file: {target.name}"
                    )

                except Exception as exc:

                    print(
                        f"   ⚠️ Could not delete "
                        f"{target.name}: {exc}"
                    )

            except Exception as exc:

                print(
                    f"❌ Processing failed "
                    f"for {month_name}: {exc}"
                )

                failed_months.append(
                    month_name
                )

                # Keep the downloaded file for
                # debugging/reprocessing.
                continue

    # ========================================================
    # Final verification
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL VERIFICATION")
    print("=" * 72)

    if OUTPUT_ZARR.exists():

        ds = xr.open_zarr(
            OUTPUT_ZARR,
            consolidated=False,
        )

        print(
            f"Dimensions: {dict(ds.sizes)}"
        )

        print(
            f"Variables: "
            f"{list(ds.data_vars)}"
        )

        print(
            f"Time range: "
            f"{ds.time.min().values} → "
            f"{ds.time.max().values}"
        )

        print(
            f"Latitude: "
            f"{float(ds.lat.min())} → "
            f"{float(ds.lat.max())}"
        )

        print(
            f"Longitude: "
            f"{float(ds.lon.min())} → "
            f"{float(ds.lon.max())}"
        )

        ds.close()

    else:

        print(
            "⚠️ No Zarr dataset was created."
        )

    print()
    print("=" * 72)
    print("DOWNLOAD SUMMARY")
    print("=" * 72)

    print(
        f"Downloaded monthly files : "
        f"{downloaded_months}"
    )

    print(
        f"Processed monthly files  : "
        f"{processed_months}"
    )

    print(
        f"Failed months            : "
        f"{len(failed_months)}"
    )

    if failed_months:

        print(
            "\nFailed months:"
        )

        for month in failed_months:
            print(f"  - {month}")

        with open(
            "failed_era5_land_months.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "\n".join(failed_months)
            )

    print()
    print(
        f"📁 Final dataset: "
        f"{OUTPUT_ZARR}"
    )

    print()
    print("✅ Finished.")


if __name__ == "__main__":
    main()
