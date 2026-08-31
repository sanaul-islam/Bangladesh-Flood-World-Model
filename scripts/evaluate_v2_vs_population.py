
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT / "src"
    ),
)


from flood_world_model.models.world_model_v2 import (
    FloodWorldModelV2,
)

from flood_world_model.datasets.multihorizon import (
    MultiHorizonFloodDataset,
)

from flood_world_model.datasets.v2_population import (
    V2PopulationDataset,
)


DYNAMIC_PATH = (
    PROJECT_ROOT
    / "data/features/dynamic_core_v2.zarr"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "data/features/static_v3.zarr"
)


V2_NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/features/training_v3/v2_normalization.json"
)

V2_POPULATION_NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/features/training_v3/"
    "v2_population_normalization.json"
)


V2_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/world_model_v2_best.pt"
)

V2_POPULATION_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/"
    "world_model_v2_population_best.pt"
)


METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/"
    "v2_vs_population_test.json"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/"
    "v2_vs_population_test.nc"
)


HISTORY_LENGTH = 14
HORIZON = 7
BATCH_SIZE = 1


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_split_indices(
    total_time: int,
) -> tuple[int, int, int]:
    train_end = int(
        total_time * 0.70
    )

    validation_end = int(
        total_time * 0.85
    )

    return (
        train_end,
        validation_end,
        total_time,
    )


def get_discharge_stats(
    normalization: dict,
) -> tuple[float, float]:
    stats = normalization[
        "river_discharge"
    ]

    mean = float(
        stats["mean"]
    )

    std = max(
        float(stats["std"]),
        1e-8,
    )

    return (
        mean,
        std,
    )


def denormalize_discharge(
    values: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    return (
        values * std + mean
    ).astype(np.float32)


def create_v2_model(
    checkpoint_path: Path,
) -> FloodWorldModelV2:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = FloodWorldModelV2(
        dynamic_channels=int(
            checkpoint[
                "dynamic_channels"
            ]
        ),
        static_channels=int(
            checkpoint[
                "static_channels"
            ]
        ),
        hidden_channels=int(
            checkpoint[
                "hidden_channels"
            ]
        ),
        horizon=int(
            checkpoint[
                "horizon"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    return model


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    valid = (
        (mask > 0.5)
        & np.isfinite(
            prediction
        )
        & np.isfinite(
            target
        )
    )

    valid_count = int(
        valid.sum()
    )

    if valid_count == 0:
        return {
            "valid_cells": 0,
            "valid_fraction": 0.0,
            "mae_m3_s": None,
            "rmse_m3_s": None,
            "bias_m3_s": None,
            "correlation": None,
        }

    p = prediction[
        valid
    ].astype(
        np.float64
    )

    y = target[
        valid
    ].astype(
        np.float64
    )

    error = p - y

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    bias = float(
        np.mean(
            error
        )
    )

    if (
        len(p) > 1
        and np.std(p) > 0
        and np.std(y) > 0
    ):
        correlation = float(
            np.corrcoef(
                p,
                y,
            )[0, 1]
        )
    else:
        correlation = None

    return {
        "valid_cells": valid_count,
        "valid_fraction": float(
            valid.mean()
        ),
        "mae_m3_s": mae,
        "rmse_m3_s": rmse,
        "bias_m3_s": bias,
        "correlation": correlation,
    }


def skill(
    model_rmse: float | None,
    baseline_rmse: float | None,
) -> float | None:
    if (
        model_rmse is None
        or baseline_rmse is None
        or baseline_rmse == 0
    ):
        return None

    return float(
        1.0
        - model_rmse
        / baseline_rmse
    )


def collect_dataset(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    predictions = []
    targets = []
    masks = []
    initial_discharges = []

    total = len(loader)

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            history = batch[
                "history"
            ].to(device)

            static = batch[
                "static"
            ].to(device)

            future_forcing = batch[
                "future_forcing"
            ].to(device)

            initial_discharge = batch[
                "initial_discharge"
            ].to(device)

            target = batch[
                "target"
            ].to(device)

            output = model(
                history=history,
                static=static,
                future_forcing=future_forcing,
                initial_discharge=initial_discharge,
                target_discharge=None,
                teacher_forcing_ratio=0.0,
            )

            if isinstance(
                output,
                (tuple, list),
            ):
                output = output[0]

            predictions.append(
                output.cpu().numpy()
            )

            targets.append(
                target.cpu().numpy()
            )

            masks.append(
                batch[
                    "mask"
                ].numpy()
            )

            initial_discharges.append(
                initial_discharge.cpu().numpy()
            )

            if (
                batch_index % 100 == 0
                or batch_index == total
            ):
                print(
                    f"Evaluated: {batch_index}/{total}"
                )

    return (
        np.concatenate(
            predictions,
            axis=0,
        ),
        np.concatenate(
            targets,
            axis=0,
        ),
        np.concatenate(
            masks,
            axis=0,
        ),
        np.concatenate(
            initial_discharges,
            axis=0,
        ),
    )


def main() -> None:
    print("=" * 80)
    print(
        "BANGLADESH FLOOD WORLD MODEL"
    )
    print(
        "V2 vs V2-POPULATION vs PERSISTENCE"
    )
    print(
        "7-DAY HELD-OUT AUTOREGRESSIVE TEST"
    )
    print("=" * 80)

    required = [
        DYNAMIC_PATH,
        STATIC_PATH,
        V2_NORMALIZATION_PATH,
        V2_POPULATION_NORMALIZATION_PATH,
        V2_CHECKPOINT_PATH,
        V2_POPULATION_CHECKPOINT_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    v2_normalization = load_json(
        V2_NORMALIZATION_PATH
    )

    population_normalization = load_json(
        V2_POPULATION_NORMALIZATION_PATH
    )

    v2_mean, v2_std = (
        get_discharge_stats(
            v2_normalization
        )
    )

    pop_mean, pop_std = (
        get_discharge_stats(
            population_normalization
        )
    )

    if not np.isclose(
        v2_mean,
        pop_mean,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "V2 and V2-Population discharge means differ."
        )

    if not np.isclose(
        v2_std,
        pop_std,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "V2 and V2-Population discharge std values differ."
        )

    discharge_mean = v2_mean
    discharge_std = v2_std

    print(
        f"Discharge normalization mean: {discharge_mean}"
    )

    print(
        f"Discharge normalization std: {discharge_std}"
    )

    ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    total_time = ds.sizes[
        "time"
    ]

    train_end, validation_end, test_end = (
        get_split_indices(
            total_time
        )
    )

    test_start_date = str(
        ds.time.values[
            validation_end
        ]
    )

    test_end_date = str(
        ds.time.values[
            test_end - 1
        ]
    )

    lat = ds.lat.values
    lon = ds.lon.values

    ds.close()

    print(
        f"Total time steps: {total_time}"
    )

    print(
        f"Train: 0 -> {train_end}"
    )

    print(
        f"Validation: {train_end} -> {validation_end}"
    )

    print(
        f"Test: {validation_end} -> {test_end}"
    )

    print(
        f"Test start date: {test_start_date}"
    )

    print(
        f"Test end date: {test_end_date}"
    )

    print()
    print(
        "Building V2 test dataset..."
    )

    v2_dataset = MultiHorizonFloodDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=V2_NORMALIZATION_PATH,
        start_index=validation_end,
        end_index=test_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print(
        f"V2 samples: {len(v2_dataset)}"
    )

    print()
    print(
        "Building V2-Population test dataset..."
    )

    population_dataset = V2PopulationDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=V2_POPULATION_NORMALIZATION_PATH,
        start_index=validation_end,
        end_index=test_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print(
        f"V2-Population samples: {len(population_dataset)}"
    )

    if (
        len(v2_dataset)
        != len(population_dataset)
    ):
        raise RuntimeError(
            "V2 and V2-Population test sample counts differ."
        )

    # -------------------------------------------------------------------------
    # Check that both datasets represent exactly the same target dates.
    # -------------------------------------------------------------------------

    v2_indices = (
        v2_dataset.indices
    )

    population_indices = (
        population_dataset.indices
    )

    if not np.array_equal(
        v2_indices,
        population_indices,
    ):
        raise RuntimeError(
            "V2 and V2-Population test indices differ."
        )

    print(
        "Test sample indices match."
    )

    # -------------------------------------------------------------------------
    # Load models.
    # -------------------------------------------------------------------------

    print()
    print(
        "Loading V2..."
    )

    v2_model = create_v2_model(
        V2_CHECKPOINT_PATH
    )

    print(
        "Loading V2-Population..."
    )

    population_model = create_v2_model(
        V2_POPULATION_CHECKPOINT_PATH
    )

    v2_parameter_count = sum(
        parameter.numel()
        for parameter in v2_model.parameters()
        if parameter.requires_grad
    )

    population_parameter_count = sum(
        parameter.numel()
        for parameter in population_model.parameters()
        if parameter.requires_grad
    )

    print(
        f"V2 parameters: {v2_parameter_count:,}"
    )

    print(
        f"V2-Population parameters: {population_parameter_count:,}"
    )

    device = torch.device(
        "cpu"
    )

    v2_loader = DataLoader(
        v2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    population_loader = DataLoader(
        population_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # -------------------------------------------------------------------------
    # V2 prediction.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "RUNNING V2"
    )
    print("=" * 80)

    (
        v2_predictions,
        v2_targets,
        v2_masks,
        v2_initial_discharges,
    ) = collect_dataset(
        v2_loader,
        v2_model,
        device,
    )

    # -------------------------------------------------------------------------
    # V2-Population prediction.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "RUNNING V2-POPULATION"
    )
    print("=" * 80)

    (
        population_predictions,
        population_targets,
        population_masks,
        population_initial_discharges,
    ) = collect_dataset(
        population_loader,
        population_model,
        device,
    )

    # -------------------------------------------------------------------------
    # Verify targets and masks correspond.
    # -------------------------------------------------------------------------

    if not np.allclose(
        v2_targets,
        population_targets,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "V2 and V2-Population targets do not match."
        )

    if not np.array_equal(
        v2_masks,
        population_masks,
    ):
        raise RuntimeError(
            "V2 and V2-Population evaluation masks do not match."
        )

    if not np.allclose(
        v2_initial_discharges,
        population_initial_discharges,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "V2 and V2-Population initial discharge states do not match."
        )

    targets_physical = (
        denormalize_discharge(
            v2_targets,
            discharge_mean,
            discharge_std,
        )
    )

    v2_physical = (
        denormalize_discharge(
            v2_predictions,
            discharge_mean,
            discharge_std,
        )
    )

    population_physical = (
        denormalize_discharge(
            population_predictions,
            discharge_mean,
            discharge_std,
        )
    )

    initial_discharge_physical = (
        denormalize_discharge(
            v2_initial_discharges,
            discharge_mean,
            discharge_std,
        )
    )

    v2_physical = np.maximum(
        v2_physical,
        0.0,
    )

    population_physical = np.maximum(
        population_physical,
        0.0,
    )

    targets_physical = np.maximum(
        targets_physical,
        0.0,
    )

    persistence_physical = np.repeat(
        initial_discharge_physical[:, None, ...],
        HORIZON,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Metrics.
    # -------------------------------------------------------------------------

    lead_results = []

    print()
    print("=" * 80)
    print(
        "LEAD-BY-LEAD COMPARISON"
    )
    print("=" * 80)

    for lead in range(
        HORIZON
    ):
        mask = v2_masks[
            :,
            lead,
        ]

        target = targets_physical[
            :,
            lead,
        ]

        v2_prediction = (
            v2_physical[
                :,
                lead,
            ]
        )

        population_prediction = (
            population_physical[
                :,
                lead,
            ]
        )

        persistence_prediction = (
            persistence_physical[
                :,
                lead,
            ]
        )

        v2_metrics = calculate_metrics(
            v2_prediction,
            target,
            mask,
        )

        population_metrics = calculate_metrics(
            population_prediction,
            target,
            mask,
        )

        persistence_metrics = calculate_metrics(
            persistence_prediction,
            target,
            mask,
        )

        v2_skill = skill(
            v2_metrics[
                "rmse_m3_s"
            ],
            persistence_metrics[
                "rmse_m3_s"
            ],
        )

        population_skill = skill(
            population_metrics[
                "rmse_m3_s"
            ],
            persistence_metrics[
                "rmse_m3_s"
            ],
        )

        population_vs_v2 = skill(
            population_metrics[
                "rmse_m3_s"
            ],
            v2_metrics[
                "rmse_m3_s"
            ],
        )

        result = {
            "lead_day": lead + 1,
            "v2": v2_metrics,
            "v2_population": population_metrics,
            "persistence": persistence_metrics,
            "v2_skill_vs_persistence": v2_skill,
            "v2_population_skill_vs_persistence": population_skill,
            "v2_population_skill_vs_v2": population_vs_v2,
        }

        lead_results.append(
            result
        )

        print("-" * 80)
        print(
            f"DAY {lead + 1}"
        )

        print(
            f"V2 RMSE: {v2_metrics['rmse_m3_s']:.3f} m3/s"
        )

        print(
            f"V2-Population RMSE: {population_metrics['rmse_m3_s']:.3f} m3/s"
        )

        print(
            f"Persistence RMSE: {persistence_metrics['rmse_m3_s']:.3f} m3/s"
        )

        print(
            f"V2 skill: {v2_skill * 100.0:.2f}%"
        )

        print(
            f"V2-Population skill: {population_skill * 100.0:.2f}%"
        )

        print(
            f"V2-Population vs V2 improvement: {population_vs_v2 * 100.0:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Pooled metrics.
    # -------------------------------------------------------------------------

    pooled_v2 = calculate_metrics(
        v2_physical,
        targets_physical,
        v2_masks,
    )

    pooled_population = calculate_metrics(
        population_physical,
        targets_physical,
        v2_masks,
    )

    pooled_persistence = calculate_metrics(
        persistence_physical,
        targets_physical,
        v2_masks,
    )

    pooled_v2_skill = skill(
        pooled_v2[
            "rmse_m3_s"
        ],
        pooled_persistence[
            "rmse_m3_s"
        ],
    )

    pooled_population_skill = skill(
        pooled_population[
            "rmse_m3_s"
        ],
        pooled_persistence[
            "rmse_m3_s"
        ],
    )

    pooled_population_vs_v2 = skill(
        pooled_population[
            "rmse_m3_s"
        ],
        pooled_v2[
            "rmse_m3_s"
        ],
    )

    print()
    print("=" * 80)
    print(
        "POOLED 7-DAY RESULTS"
    )
    print("=" * 80)

    print(
        f"V2 MAE: {pooled_v2['mae_m3_s']:.3f} m3/s"
    )

    print(
        f"V2 RMSE: {pooled_v2['rmse_m3_s']:.3f} m3/s"
    )

    print(
        f"V2 Bias: {pooled_v2['bias_m3_s']:.3f} m3/s"
    )

    print(
        f"V2 Correlation: {pooled_v2['correlation']}"
    )

    print(
        f"V2 Skill vs Persistence: {pooled_v2_skill * 100.0:.2f}%"
    )

    print()
    print(
        f"V2-Population MAE: {pooled_population['mae_m3_s']:.3f} m3/s"
    )

    print(
        f"V2-Population RMSE: {pooled_population['rmse_m3_s']:.3f} m3/s"
    )

    print(
        f"V2-Population Bias: {pooled_population['bias_m3_s']:.3f} m3/s"
    )

    print(
        f"V2-Population Correlation: {pooled_population['correlation']}"
    )

    print(
        f"V2-Population Skill vs Persistence: {pooled_population_skill * 100.0:.2f}%"
    )

    print()
    print(
        f"V2-Population improvement vs V2: {pooled_population_vs_v2 * 100.0:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Save prediction NetCDF.
    # -------------------------------------------------------------------------

    FORECAST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_count = (
        v2_physical.shape[0]
    )

    forecast_ds = xr.Dataset(
        {
            "v2_predicted_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                v2_physical,
            ),
            "v2_population_predicted_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                population_physical,
            ),
            "persistence_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                persistence_physical,
            ),
            "actual_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                targets_physical,
            ),
            "evaluation_mask": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                v2_masks.astype(
                    np.float32
                ),
            ),
        },
        coords={
            "sample": np.arange(
                sample_count
            ),
            "forecast_day": np.arange(
                1,
                HORIZON + 1,
            ),
            "lat": lat,
            "lon": lon,
        },
    )

    forecast_ds[
        "v2_predicted_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds[
        "v2_population_predicted_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds[
        "persistence_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds[
        "actual_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds.attrs[
        "test_period"
    ] = (
        f"{test_start_date} -> {test_end_date}"
    )

    forecast_ds.attrs[
        "evaluation_mask"
    ] = (
        "glofas_discharge_valid_t × river_mask"
    )

    forecast_ds.attrs[
        "forcing"
    ] = (
        "observed future precipitation"
    )

    forecast_ds.attrs[
        "purpose"
    ] = (
        "V2 vs V2-Population controlled comparison"
    )

    forecast_ds.to_netcdf(
        FORECAST_PATH
    )

    forecast_ds.close()

    # -------------------------------------------------------------------------
    # Save JSON.
    # -------------------------------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "comparison": (
            "V2 vs V2-Population vs persistence"
        ),
        "test_start_index": validation_end,
        "test_end_index": test_end,
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
        "history_length": HISTORY_LENGTH,
        "horizon": HORIZON,
        "test_samples": sample_count,
        "evaluation_mask": (
            "glofas_discharge_valid_t × river_mask"
        ),
        "forcing": (
            "observed future precipitation"
        ),
        "models": {
            "v2": {
                "checkpoint": str(
                    V2_CHECKPOINT_PATH
                ),
                "parameters": v2_parameter_count,
                "static_channels": 11,
            },
            "v2_population": {
                "checkpoint": str(
                    V2_POPULATION_CHECKPOINT_PATH
                ),
                "parameters": population_parameter_count,
                "static_channels": 12,
                "population_feature": True,
            },
        },
        "lead_metrics": lead_results,
        "pooled_v2": pooled_v2,
        "pooled_v2_population": pooled_population,
        "pooled_persistence": pooled_persistence,
        "pooled_v2_skill_vs_persistence": pooled_v2_skill,
        "pooled_v2_population_skill_vs_persistence": pooled_population_skill,
        "pooled_v2_population_skill_vs_v2": pooled_population_vs_v2,
        "decision": {
            "v2_population_better_than_v2": bool(
                pooled_population[
                    "rmse_m3_s"
                ]
                < pooled_v2[
                    "rmse_m3_s"
                ]
            ),
            "v2_population_beats_persistence": bool(
                pooled_population[
                    "rmse_m3_s"
                ]
                < pooled_persistence[
                    "rmse_m3_s"
                ]
            ),
        },
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "COMPARISON COMPLETE"
    )
    print("=" * 80)

    print(
        f"Forecast file: {FORECAST_PATH}"
    )

    print(
        f"Metrics file: {METRICS_PATH}"
    )

    if (
        pooled_population[
            "rmse_m3_s"
        ]
        < pooled_v2[
            "rmse_m3_s"
        ]
    ):
        print(
            "RESULT: V2-Population beats V2 on pooled RMSE."
        )
    else:
        print(
            "RESULT: V2-Population does NOT beat V2 on pooled RMSE."
        )


if __name__ == "__main__":
    main()
