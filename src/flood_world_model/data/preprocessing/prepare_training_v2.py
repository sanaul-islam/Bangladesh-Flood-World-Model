from __future__ import annotations

import gc
import json

import numpy as np
import xarray as xr

from flood_world_model.utils.paths import FEATURES_DIR, PROJECT_ROOT


print("=" * 80)
print("BANGLADESH FLOOD WORLD MODEL")
print("TRAINING PREPARATION V3")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

DYNAMIC_PATH = FEATURES_DIR / "dynamic_core_v2.zarr"
STATIC_PATH = FEATURES_DIR / "static_v3.zarr"
STATIC_MASK_PATH = FEATURES_DIR / "static_masks_v3.zarr"
OUTPUT_DIR = FEATURES_DIR / "training_v3"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# WINDOW
# ============================================================

HISTORY_DAYS = 14
FORECAST_DAYS = 1


# ============================================================
# SPLITS
# ============================================================

TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"

VAL_START = "2022-01-01"
VAL_END = "2022-12-31"

TEST_START = "2023-01-01"
TEST_END = "2025-12-31"

LIVE_START = "2026-01-01"
LIVE_END = "2026-06-01"


# ============================================================
# DYNAMIC INPUTS
# ============================================================
#
# We include the GloFAS validity mask as an input because
# river discharge contains structural NaNs.
#
# The mask tells the model:
#
#   1 = discharge field exists
#   0 = discharge unavailable here
# ============================================================

DYNAMIC_INPUTS = [
    "precipitation",
    "precip_3d",
    "precip_7d",
    "precip_log1p",
    "precip_missing",
    "river_discharge",
    "glofas_discharge_valid_t",
]


# ============================================================
# TARGET
# ============================================================

TARGETS = [
    "river_discharge",
]


# ============================================================
# STATIC INPUTS
# ============================================================

STATIC_INPUTS = [
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
    "land_mask",
]


# ============================================================
# HELPERS
# ============================================================

def require_dataset(
    path: Path,
    name: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def check_variables(
    ds: xr.Dataset,
    variables: list[str],
    name: str,
) -> None:

    missing = [
        v
        for v in variables
        if v not in ds.data_vars
    ]

    if missing:

        raise RuntimeError(
            f"{name} is missing:\n"
            + "\n".join(
                f"  - {v}"
                for v in missing
            )
            + "\n\nAvailable:\n"
            + "\n".join(
                f"  - {v}"
                for v in ds.data_vars
            )
        )


def compute_statistics(
    da: xr.DataArray,
) -> dict:

    """
    Statistics are computed lazily.
    Invalid/NaN values are ignored.
    """

    mean = float(
        da.mean(
            skipna=True
        ).compute()
    )

    std = float(
        da.std(
            skipna=True
        ).compute()
    )

    minimum = float(
        da.min(
            skipna=True
        ).compute()
    )

    maximum = float(
        da.max(
            skipna=True
        ).compute()
    )

    if not np.isfinite(
        mean
    ):
        mean = 0.0

    if (
        not np.isfinite(std)
        or std < 1e-8
    ):
        std = 1.0

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
    }


def get_split_indices(
    target_dates: np.ndarray,
    starts: np.ndarray,
    start_date: str,
    end_date: str,
) -> np.ndarray:

    mask = (
        target_dates
        >= np.datetime64(
            start_date,
            "D",
        )
    ) & (
        target_dates
        <= np.datetime64(
            end_date,
            "D",
        )
    )

    return starts[
        mask
    ]


# ============================================================
# 1. CHECK FILES
# ============================================================

require_dataset(
    DYNAMIC_PATH,
    "Dynamic V2",
)

require_dataset(
    STATIC_PATH,
    "Static V3",
)

require_dataset(
    STATIC_MASK_PATH,
    "Static masks V3",
)


# ============================================================
# 2. OPEN ZARRS LAZILY
# ============================================================

print("\nLoading dynamic...")

dynamic = xr.open_zarr(
    DYNAMIC_PATH,
    consolidated=True,
)

print(
    "Dynamic:",
    dict(dynamic.sizes),
)

print(
    "Dynamic variables:",
    list(dynamic.data_vars),
)


print("\nLoading static...")

static = xr.open_zarr(
    STATIC_PATH,
    consolidated=True,
)

print(
    "Static:",
    dict(static.sizes),
)


print("\nLoading static masks...")

static_masks = xr.open_zarr(
    STATIC_MASK_PATH,
    consolidated=True,
)

print(
    "Static masks:",
    list(static_masks.data_vars),
)


# ============================================================
# 3. VARIABLE VALIDATION
# ============================================================

check_variables(
    dynamic,
    DYNAMIC_INPUTS,
    "dynamic_core_v2.zarr",
)

check_variables(
    dynamic,
    TARGETS,
    "dynamic_core_v2.zarr",
)

check_variables(
    static,
    STATIC_INPUTS,
    "static_v3.zarr",
)


# ============================================================
# 4. GRID VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("GRID VALIDATION")
print("=" * 80)

if not np.allclose(
    dynamic.lat.values,
    static.lat.values,
):
    raise RuntimeError(
        "Dynamic/static latitude mismatch."
    )

if not np.allclose(
    dynamic.lon.values,
    static.lon.values,
):
    raise RuntimeError(
        "Dynamic/static longitude mismatch."
    )

print(
    f"✅ Grid = "
    f"{dynamic.sizes['lat']} × "
    f"{dynamic.sizes['lon']}"
)


# ============================================================
# 5. TIME VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("TIME VALIDATION")
print("=" * 80)

times = dynamic.time.values.astype(
    "datetime64[D]"
)

print(
    f"Start: {times[0]}"
)

print(
    f"End  : {times[-1]}"
)

print(
    f"Count: {len(times):,}"
)

time_diff = np.diff(
    times
)

bad_gaps = (
    time_diff
    != np.timedelta64(
        1,
        "D",
    )
)

print(
    f"Non-daily gaps: "
    f"{int(bad_gaps.sum())}"
)

if bad_gaps.any():

    print(
        "⚠️ Time gaps exist."
    )

else:

    print(
        "✅ Daily sequence."
    )


# ============================================================
# 6. BUILD WINDOWS
# ============================================================

max_start = (
    len(times)
    - HISTORY_DAYS
    - FORECAST_DAYS
    + 1
)

if max_start <= 0:

    raise RuntimeError(
        "Not enough data for temporal windows."
    )

starts = np.arange(
    max_start,
    dtype=np.int64,
)

target_dates = times[
    starts + HISTORY_DAYS
]


train_indices = get_split_indices(
    target_dates,
    starts,
    TRAIN_START,
    TRAIN_END,
)

val_indices = get_split_indices(
    target_dates,
    starts,
    VAL_START,
    VAL_END,
)

test_indices = get_split_indices(
    target_dates,
    starts,
    TEST_START,
    TEST_END,
)

live_indices = get_split_indices(
    target_dates,
    starts,
    LIVE_START,
    LIVE_END,
)


print("\n" + "=" * 80)
print("WINDOW SPLITS")
print("=" * 80)

print(
    f"Train: {len(train_indices):,}"
)

print(
    f"Val  : {len(val_indices):,}"
)

print(
    f"Test : {len(test_indices):,}"
)

print(
    f"Live : {len(live_indices):,}"
)


# ============================================================
# 7. SAVE INDICES
# ============================================================

np.save(
    OUTPUT_DIR / "train_indices.npy",
    train_indices,
)

np.save(
    OUTPUT_DIR / "val_indices.npy",
    val_indices,
)

np.save(
    OUTPUT_DIR / "test_indices.npy",
    test_indices,
)

np.save(
    OUTPUT_DIR / "live_indices.npy",
    live_indices,
)


# ============================================================
# 8. NORMALIZATION
# ============================================================

print("\n" + "=" * 80)
print("TRAIN-ONLY NORMALIZATION")
print("=" * 80)

normalization = {}


# ============================================================
# DYNAMIC NORMALIZATION
# ============================================================

for variable in DYNAMIC_INPUTS:

    print(
        f"\nDynamic: {variable}"
    )

    # Binary masks don't need standardization.

    if variable in [
        "precip_missing",
        "glofas_discharge_valid_t",
    ]:

        normalization[
            variable
        ] = {
            "type": "binary",
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 1.0,
        }

        print(
            "  type: binary"
        )

        continue


    da = dynamic[
        variable
    ].sel(
        time=slice(
            TRAIN_START,
            TRAIN_END,
        )
    )

    stats = compute_statistics(
        da
    )

    normalization[
        variable
    ] = {
        **stats,
        "type": "standard",
    }

    print(
        f"  mean={stats['mean']:.6f}"
    )

    print(
        f"  std ={stats['std']:.6f}"
    )

    print(
        f"  min ={stats['min']:.6f}"
    )

    print(
        f"  max ={stats['max']:.6f}"
    )

    del da

    gc.collect()


# ============================================================
# STATIC NORMALIZATION
# ============================================================

print("\n" + "=" * 80)
print("STATIC NORMALIZATION")
print("=" * 80)

for variable in STATIC_INPUTS:

    print(
        f"\nStatic: {variable}"
    )

    da = static[
        variable
    ]

    if variable == "landcover":

        normalization[
            f"static_{variable}"
        ] = {
            "type": "categorical",
            "mean": 0.0,
            "std": 1.0,
            "min": float(
                da.min().compute()
            ),
            "max": float(
                da.max().compute()
            ),
        }

        print(
            "  type: categorical"
        )

        continue


    if variable in [
        "river_mask",
        "land_mask",
    ]:

        normalization[
            f"static_{variable}"
        ] = {
            "type": "binary",
            "mean": 0.0,
            "std": 1.0,
            "min": float(
                da.min().compute()
            ),
            "max": float(
                da.max().compute()
            ),
        }

        print(
            "  type: binary"
        )

        continue


    stats = compute_statistics(
        da
    )

    normalization[
        f"static_{variable}"
    ] = {
        **stats,
        "type": "standard",
    }

    print(
        f"  mean={stats['mean']:.6f}"
    )

    print(
        f"  std ={stats['std']:.6f}"
    )

    del da

    gc.collect()


# ============================================================
# 9. SAVE NORMALIZATION
# ============================================================

with (
    OUTPUT_DIR
    / "normalization.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        normalization,
        f,
        indent=2,
    )

print(
    "\n✅ normalization.json saved."
)


# ============================================================
# 10. SAVE DATASET INFO
# ============================================================

info = {

    "version":
        "training_v3",

    "history_days":
        HISTORY_DAYS,

    "forecast_days":
        FORECAST_DAYS,

    "grid": {
        "lat":
            int(dynamic.sizes["lat"]),
        "lon":
            int(dynamic.sizes["lon"]),
    },

    "dynamic_inputs":
        DYNAMIC_INPUTS,

    "targets":
        TARGETS,

    "static_inputs":
        STATIC_INPUTS,

    "splits": {

        "train":
            [TRAIN_START, TRAIN_END],

        "validation":
            [VAL_START, VAL_END],

        "test":
            [TEST_START, TEST_END],

        "live":
            [LIVE_START, LIVE_END],
    },

    "sample_counts": {

        "train":
            int(len(train_indices)),

        "validation":
            int(len(val_indices)),

        "test":
            int(len(test_indices)),

        "live":
            int(len(live_indices)),
    },

    "time_range": [
        str(times[0]),
        str(times[-1]),
    ],

    "missing_value_policy": (
        "Source NaNs are preserved in Zarr. "
        "Training Dataset converts invalid input "
        "values to zero after normalization and "
        "uses explicit validity masks for targets."
    ),
}


with (
    OUTPUT_DIR
    / "dataset_info.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        info,
        f,
        indent=2,
    )


# ============================================================
# 11. EXAMPLE WINDOW
# ============================================================

print("\n" + "=" * 80)
print("EXAMPLE WINDOW")
print("=" * 80)

start = int(
    train_indices[0]
)

x_start = start

x_end = (
    start
    + HISTORY_DAYS
)

y_start = x_end

y_end = (
    y_start
    + FORECAST_DAYS
)

print(
    "Input:",
    times[x_start],
    "→",
    times[x_end - 1],
)

print(
    "Target:",
    times[y_start],
    "→",
    times[y_end - 1],
)


# ============================================================
# 12. EXAMPLE INPUT
# ============================================================
#
# We intentionally DO NOT require raw river_discharge
# to be finite.
#
# NaNs are expected outside the GloFAS discharge domain.
# ============================================================

example = dynamic[
    DYNAMIC_INPUTS
].isel(
    time=slice(
        x_start,
        x_end,
    )
).compute()


for variable in DYNAMIC_INPUTS:

    values = example[
        variable
    ].values

    nan_count = int(
        np.isnan(values).sum()
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    print(
        f"{variable:35s}"
        f"NaN={nan_count:,} "
        f"Inf={inf_count:,}"
    )


# ============================================================
# 13. IMPORTANT: EXPECTED NaNs
# ============================================================

river_input = (
    example[
        "river_discharge"
    ].values
)

river_mask_input = (
    example[
        "glofas_discharge_valid_t"
    ].values
)


river_valid = np.isfinite(
    river_input
)

mask_positive = (
    river_mask_input >= 0.5
)


# Where the GloFAS mask says valid, discharge should
# normally be finite.
bad_valid = (
    mask_positive
    & (~river_valid)
)

print(
    "\nDischarge-mask consistency:"
)

print(
    f"Mask valid cells: "
    f"{mask_positive.sum():,}"
)

print(
    f"Discharge finite: "
    f"{river_valid.sum():,}"
)

print(
    f"Mask says valid but discharge NaN: "
    f"{bad_valid.sum():,}"
)

if bad_valid.any():

    print(
        "⚠️ Some GloFAS-valid cells have "
        "missing discharge."
    )

else:

    print(
        "✅ Mask/discharge consistency looks good."
    )


# ============================================================
# 14. TARGET CHECK
# ============================================================

print("\n" + "=" * 80)
print("TARGET CHECK")
print("=" * 80)

target = dynamic[
    TARGETS
].isel(
    time=slice(
        y_start,
        y_end,
    )
).compute()

target_discharge = (
    target[
        "river_discharge"
    ].values
)

target_valid = np.isfinite(
    target_discharge
)

target_count = (
    target_valid.size
)

print(
    f"Target finite: "
    f"{target_valid.sum():,}/"
    f"{target_count:,}"
)

print(
    f"Target valid fraction: "
    f"{target_valid.mean():.4%}"
)

if not target_valid.any():

    raise RuntimeError(
        "Example target contains no valid discharge."
    )


# ============================================================
# 15. STATIC CHECK
# ============================================================

print("\n" + "=" * 80)
print("STATIC CHECK")
print("=" * 80)

for variable in STATIC_INPUTS:

    values = (
        static[
            variable
        ].values
    )

    finite = np.isfinite(
        values
    )

    print(
        f"{variable:35s}"
        f"finite={finite.sum():4d}/"
        f"{finite.size:4d}"
    )

    if not finite.all():

        raise RuntimeError(
            f"Static variable {variable} "
            "still contains NaN/Inf."
        )


# ============================================================
# 16. CHECK MASK DATASET
# ============================================================

print("\n" + "=" * 80)
print("STATIC MASK CHECK")
print("=" * 80)

for variable in static_masks.data_vars:

    values = (
        static_masks[
            variable
        ].values
    )

    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Static mask {variable} "
            "contains NaN/Inf."
        )

    print(
        f"✅ {variable}"
    )


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary = {

    "status":
        "PASS",

    "history_days":
        HISTORY_DAYS,

    "forecast_days":
        FORECAST_DAYS,

    "dynamic_inputs":
        DYNAMIC_INPUTS,

    "targets":
        TARGETS,

    "static_inputs":
        STATIC_INPUTS,

    "raw_input_nan_policy":
        (
            "River discharge NaNs are expected outside "
            "the GloFAS discharge domain and are not "
            "treated as source corruption."
        ),

    "target_mask":
        "glofas_discharge_valid_t",

    "grid": [
        int(dynamic.sizes["lat"]),
        int(dynamic.sizes["lon"]),
    ],
}


with (
    OUTPUT_DIR
    / "preparation_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# CLOSE
# ============================================================

dynamic.close()
static.close()
static_masks.close()

del dynamic
del static
del static_masks
del example
del target

gc.collect()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("🎉 TRAINING PREPARATION V3 COMPLETE")
print("=" * 80)

print(
    f"Output: {OUTPUT_DIR}"
)

print(
    "\nImportant:"
)

print(
    "  River discharge NaNs were NOT treated as bad source data."
)

print(
    "  GloFAS validity is represented by:"
)

print(
    "  glofas_discharge_valid_t"
)

print(
    "\nNext:"
)

print(
    "Use training_v3 with the new PyTorch Dataset."
)