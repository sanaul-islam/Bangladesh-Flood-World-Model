from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import requests


# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path("data/raw/openmeteo_grid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bangladesh bounding box
LAT_MIN = 20.5
LAT_MAX = 26.5

LON_MIN = 88.0
LON_MAX = 92.5

STEP = 0.1

# Number of coordinates per API request.
# 25 = 5 x 5 spatial block.
CHUNK_SIZE = 25

# Your NASA dataset is daily, so download daily weather.
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
]

# Historical period
START_YEAR = 2018
END_YEAR = 2026

# API/network configuration
REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 5

# Use GMT/UTC to keep all sources aligned.
TIMEZONE = "GMT"


# ============================================================
# Build the 0.1 degree grid
# ============================================================

def make_grid():
    """
    Build the model grid.

    np.round avoids floating-point values such as
    88.99999999999999 appearing in URLs.
    """

    lats = np.round(
        np.arange(
            LAT_MIN,
            LAT_MAX + STEP / 2,
            STEP,
        ),
        4,
    )

    lons = np.round(
        np.arange(
            LON_MIN,
            LON_MAX + STEP / 2,
            STEP,
        ),
        4,
    )

    points = [
        (float(lat), float(lon))
        for lat in lats
        for lon in lons
    ]

    return lats, lons, points


# ============================================================
# Split grid into chunks
# ============================================================

def make_chunks(points, chunk_size=CHUNK_SIZE):
    return [
        points[i:i + chunk_size]
        for i in range(0, len(points), chunk_size)
    ]


# ============================================================
# Request helper
# ============================================================

def fetch_chunk(
    session: requests.Session,
    points: list[tuple[float, float]],
    start_date: str,
    end_date: str,
):
    """
    Query multiple coordinates in a single Open-Meteo
    GET request.

    Open-Meteo returns a list of location objects when
    multiple coordinates are requested.
    """

    latitudes = ",".join(
        f"{lat:.4f}"
        for lat, _ in points
    )

    longitudes = ",".join(
        f"{lon:.4f}"
        for _, lon in points
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------
            if response.status_code == 200:

                data = response.json()

                # API-level error returned as HTTP 200
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(
                        data.get(
                            "reason",
                            "Unknown Open-Meteo error",
                        )
                    )

                return data

            # ------------------------------------------------
            # Rate limiting
            # ------------------------------------------------
            if response.status_code == 429:

                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))

                print(
                    f"429 rate limit. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)
                continue

            # ------------------------------------------------
            # Server errors
            # ------------------------------------------------
            if response.status_code >= 500:

                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))

                print(
                    f"Server error {response.status_code}. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)
                continue

            # ------------------------------------------------
            # Other HTTP errors
            # ------------------------------------------------
            raise RuntimeError(
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:

            wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))

            print(
                f"Network error on attempt "
                f"{attempt}/{MAX_RETRIES}: {exc}"
            )

            if attempt < MAX_RETRIES:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

        except Exception:
            # Don't retry malformed requests/API errors endlessly.
            raise

    raise RuntimeError(
        "Request failed after maximum retries."
    )


# ============================================================
# Validate response
# ============================================================

def validate_response(data, expected_points):
    """
    Basic structural validation before saving.
    """

    if not isinstance(data, list):
        data = [data]

    if len(data) != len(expected_points):
        raise ValueError(
            f"Expected {len(expected_points)} locations, "
            f"received {len(data)}."
        )

    for item, (expected_lat, expected_lon) in zip(
        data,
        expected_points,
    ):

        if "daily" not in item:
            raise ValueError(
                "Missing 'daily' data in API response."
            )

        daily = item["daily"]

        if "time" not in daily:
            raise ValueError(
                "Missing daily time array."
            )

        if "latitude" not in item:
            raise ValueError(
                "Missing latitude in response."
            )

        if "longitude" not in item:
            raise ValueError(
                "Missing longitude in response."
            )

    return data


# ============================================================
# Main downloader
# ============================================================

def main():

    print("=" * 70)
    print("Open-Meteo Daily Grid Downloader")
    print("=" * 70)

    lats, lons, points = make_grid()

    chunks = make_chunks(points)

    print(
        f"Grid latitude points : {len(lats)}"
    )

    print(
        f"Grid longitude points: {len(lons)}"
    )

    print(
        f"Total grid points    : {len(points)}"
    )

    print(
        f"Requests per year    : {len(chunks)}"
    )

    print(
        f"Years                 : "
        f"{START_YEAR}-{END_YEAR}"
    )

    print(
        f"Daily variables       : "
        f"{DAILY_VARIABLES}"
    )

    print()

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Bangladesh-Flood-World-Model/1.0 "
                "(research/student project)"
            )
        }
    )

    total_requests = 0
    successful_requests = 0
    failed_requests = 0

    # ========================================================
    # Year loop
    # ========================================================

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        print()
        print("=" * 70)
        print(f"YEAR {year}")
        print("=" * 70)

        # ====================================================
        # Chunk loop
        # ====================================================

        for chunk_index, points_chunk in enumerate(
            chunks,
            start=1,
        ):

            output_file = (
                OUTPUT_DIR
                / f"openmeteo_{year}_{chunk_index:04d}.json"
            )

            # ------------------------------------------------
            # Resume support
            # ------------------------------------------------

            if (
                output_file.exists()
                and output_file.stat().st_size > 1000
            ):

                print(
                    f"[{chunk_index:04d}/{len(chunks)}] "
                    f"already exists → skip"
                )

                successful_requests += 1
                continue

            total_requests += 1

            print(
                f"[{chunk_index:04d}/{len(chunks)}] "
                f"{len(points_chunk)} locations...",
                end=" ",
                flush=True,
            )

            try:

                data = fetch_chunk(
                    session=session,
                    points=points_chunk,
                    start_date=start_date,
                    end_date=end_date,
                )

                data = validate_response(
                    data,
                    points_chunk,
                )

                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                with output_file.open(
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        data,
                        f,
                        separators=(",", ":"),
                    )

                size_mb = (
                    output_file.stat().st_size
                    / (1024 ** 2)
                )

                successful_requests += 1

                print(
                    f"✅ {size_mb:.2f} MB"
                )

            except Exception as exc:

                failed_requests += 1

                print(
                    f"❌ FAILED: {exc}"
                )

                # Don't abort the complete archive.
                # Move to next chunk.
                continue

            # Small delay to be polite to the API.
            time.sleep(0.5)

    session.close()

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Total new requests : {total_requests}"
    )

    print(
        f"Successful         : {successful_requests}"
    )

    print(
        f"Failed             : {failed_requests}"
    )

    print(
        f"Output directory   : {OUTPUT_DIR}"
    )

    print()
    print("✅ Download process finished.")


if __name__ == "__main__":
    main()
