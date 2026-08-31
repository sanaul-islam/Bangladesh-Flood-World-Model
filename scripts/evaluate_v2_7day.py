
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

from flood_world_model.datasets.multihorizon import (
    MultiHorizonFloodDataset,
)

from flood_world_model.models.world_model_v2 import (
    FloodWorldModelV2,
)


DYNAMIC_PATH = (
    PROJECT_ROOT
    / "data/features/dynamic_core_v2.zarr"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "data/features/static_v3.zarr"
)

NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/features/training_v3/v2_normalization.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/world_model_v2_best.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/world_model_v2_7day_test.json"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs/predictions/world_model_v2_7day_test.nc"
)

HISTORY_LENGTH = 14
HORIZON = 7
BATCH_SIZE = 1


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


def load_normalization() -> dict:
    with NORMALIZATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def denormalize_discharge(
    values: np.ndarray,
    normalization: dict,
) -> np.ndarray:
    stats = normalization[
        "river_discharge"
    ]

    mean = float(
        stats["mean"]
    )

    std = max(
        float(
            stats["std"]
        ),
        1e-8,
    )

    return (
        values * std + mean
    ).astype(np.float32)


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    valid = (
        (mask > 0.5)
        & np.isfinite(prediction)
        & np.isfinite(target)
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
    ].astype(np.float64)

    y = target[
        valid
    ].astype(np.float64)

    error = p - y

    mae = float(
        np.mean(
            np.abs(error)
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
        np.mean(error)
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


def calculate_skill(
    model_rmse: float | None,
    persistence_rmse: float | None,
) -> float | None:
    if (
        model_rmse is None
        or persistence_rmse is None
        or persistence_rmse == 0
    ):
        return None

    return float(
        1.0
        - (
            model_rmse
            / persistence_rmse
        )
    )


def calculate_lead_metrics(
    predictions_normalized: np.ndarray,
    targets_normalized: np.ndarray,
    masks: np.ndarray,
    persistence_normalized: np.ndarray,
    normalization: dict,
) -> list[dict]:
    predictions = denormalize_discharge(
        predictions_normalized,
        normalization,
    )

    targets = denormalize_discharge(
        targets_normalized,
        normalization,
    )

    persistence = denormalize_discharge(
        persistence_normalized,
        normalization,
    )

    results = []

    for lead in range(HORIZON):
        model_metrics = calculate_metrics(
            predictions[:, lead],
            targets[:, lead],
            masks[:, lead],
        )

        persistence_metrics = calculate_metrics(
            persistence[:, lead],
            targets[:, lead],
            masks[:, lead],
        )

        skill = calculate_skill(
            model_metrics[
                "rmse_m3_s"
            ],
            persistence_metrics[
                "rmse_m3_s"
            ],
        )

        results.append(
            {
                "lead_day": lead + 1,
                "v2": model_metrics,
                "persistence": persistence_metrics,
                "skill_vs_persistence": skill,
            }
        )

    return results


@torch.inference_mode()
def predict_v2(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    predictions = []
    targets = []
    masks = []
    initial_discharges = []

    total_batches = len(loader)

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
            or batch_index == total_batches
        ):
            print(
                f"Evaluated samples: {batch_index}/{total_batches}"
            )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    masks = np.concatenate(
        masks,
        axis=0,
    )

    initial_discharges = np.concatenate(
        initial_discharges,
        axis=0,
    )

    return (
        predictions,
        targets,
        masks,
        initial_discharges,
    )


def main() -> None:
    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL V2")
    print("7-DAY HELD-OUT AUTOREGRESSIVE TEST")
    print("=" * 80)

    required_files = [
        DYNAMIC_PATH,
        STATIC_PATH,
        NORMALIZATION_PATH,
        CHECKPOINT_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    normalization = load_normalization()

    dynamic_ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    total_time = dynamic_ds.sizes[
        "time"
    ]

    train_end, validation_end, test_end = (
        get_split_indices(
            total_time
        )
    )

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

    test_times = dynamic_ds.time.values[
        validation_end:
        test_end
    ]

    lat_values = dynamic_ds.lat.values
    lon_values = dynamic_ds.lon.values

    dynamic_ds.close()

    test_dataset = MultiHorizonFloodDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=validation_end,
        end_index=test_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    device = torch.device(
        "cpu"
    )

    model = model.to(
        device
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

    print("=" * 80)
    print("RUNNING V2 AUTOREGRESSIVE TEST")
    print("=" * 80)

    (
        predictions,
        targets,
        masks,
        initial_discharges,
    ) = predict_v2(
        model,
        test_loader,
        device,
    )

    print(
        f"Prediction shape: {predictions.shape}"
    )

    print(
        f"Target shape: {targets.shape}"
    )

    print(
        f"Mask shape: {masks.shape}"
    )

    persistence = np.repeat(
        initial_discharges[:, None, ...],
        HORIZON,
        axis=1,
    )

    lead_metrics = calculate_lead_metrics(
        predictions_normalized=predictions,
        targets_normalized=targets,
        masks=masks,
        persistence_normalized=persistence,
        normalization=normalization,
    )

    print()
    print("=" * 80)
    print("V2 LEAD-BY-LEAD RESULTS")
    print("=" * 80)

    for result in lead_metrics:
        lead = result[
            "lead_day"
        ]

        v2 = result[
            "v2"
        ]

        baseline = result[
            "persistence"
        ]

        skill = result[
            "skill_vs_persistence"
        ]

        print("-" * 80)
        print(
            f"DAY {lead}"
        )

        print(
            f"V2 MAE: {v2['mae_m3_s']:.3f} m3/s"
        )

        print(
            f"V2 RMSE: {v2['rmse_m3_s']:.3f} m3/s"
        )

        print(
            f"V2 Bias: {v2['bias_m3_s']:.3f} m3/s"
        )

        print(
            f"V2 Correlation: {v2['correlation']}"
        )

        print(
            f"Persistence MAE: {baseline['mae_m3_s']:.3f} m3/s"
        )

        print(
            f"Persistence RMSE: {baseline['rmse_m3_s']:.3f} m3/s"
        )

        print(
            f"Persistence Bias: {baseline['bias_m3_s']:.3f} m3/s"
        )

        print(
            f"Persistence Correlation: {baseline['correlation']}"
        )

        if skill is None:
            print(
                "Skill vs persistence: None"
            )
        else:
            print(
                f"Skill vs persistence: {skill * 100.0:.2f}%"
            )

    prediction_physical = denormalize_discharge(
        predictions,
        normalization,
    )

    target_physical = denormalize_discharge(
        targets,
        normalization,
    )

    persistence_physical = denormalize_discharge(
        persistence,
        normalization,
    )

    prediction_physical = np.maximum(
        prediction_physical,
        0.0,
    )

    target_physical = np.maximum(
        target_physical,
        0.0,
    )

    persistence_physical = np.maximum(
        persistence_physical,
        0.0,
    )

    pooled_v2 = calculate_metrics(
        prediction_physical,
        target_physical,
        masks,
    )

    pooled_persistence = calculate_metrics(
        persistence_physical,
        target_physical,
        masks,
    )

    pooled_skill = calculate_skill(
        pooled_v2[
            "rmse_m3_s"
        ],
        pooled_persistence[
            "rmse_m3_s"
        ],
    )

    print()
    print("=" * 80)
    print("POOLED 7-DAY RESULTS")
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
        f"Persistence MAE: {pooled_persistence['mae_m3_s']:.3f} m3/s"
    )

    print(
        f"Persistence RMSE: {pooled_persistence['rmse_m3_s']:.3f} m3/s"
    )

    print(
        f"Persistence Bias: {pooled_persistence['bias_m3_s']:.3f} m3/s"
    )

    print(
        f"Persistence Correlation: {pooled_persistence['correlation']}"
    )

    if pooled_skill is None:
        print(
            "Overall skill vs persistence: None"
        )
    else:
        print(
            f"Overall skill vs persistence: {pooled_skill * 100.0:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Save NetCDF
    # -------------------------------------------------------------------------

    FORECAST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_count = (
        prediction_physical.shape[0]
    )

    forecast_ds = xr.Dataset(
        {
            "v2_predicted_river_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                prediction_physical,
            ),
            "actual_river_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                target_physical,
            ),
            "persistence_river_discharge": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                persistence_physical,
            ),
            "evaluation_mask": (
                (
                    "sample",
                    "forecast_day",
                    "lat",
                    "lon",
                ),
                masks.astype(np.float32),
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
            "lat": lat_values,
            "lon": lon_values,
        },
    )

    forecast_ds[
        "v2_predicted_river_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds[
        "actual_river_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds[
        "persistence_river_discharge"
    ].attrs["units"] = "m3 s-1"

    forecast_ds.attrs[
        "model"
    ] = "Bangladesh Flood World Model V2"

    forecast_ds.attrs[
        "forecast_type"
    ] = "held_out_autoregressive_7day"

    forecast_ds.attrs[
        "future_forcing"
    ] = "observed_test_period_precipitation"

    forecast_ds.attrs[
        "evaluation_mask"
    ] = "glofas_discharge_valid_t × river_mask"

    forecast_ds.to_netcdf(
        FORECAST_PATH
    )

    forecast_ds.close()

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": "Bangladesh Flood World Model V2",
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "normalization": str(
            NORMALIZATION_PATH
        ),
        "dynamic_dataset": str(
            DYNAMIC_PATH
        ),
        "static_dataset": str(
            STATIC_PATH
        ),
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": HORIZON,
        "test_start_index": validation_end,
        "test_end_index": test_end,
        "test_sample_count": sample_count,
        "test_start_date": str(
            test_times[0]
        ),
        "test_end_date": str(
            test_times[-1]
        ),
        "parameter_count": parameter_count,
        "evaluation_mask": (
            "glofas_discharge_valid_t × river_mask"
        ),
        "future_forcing": (
            "observed precipitation from held-out period"
        ),
        "lead_metrics": lead_metrics,
        "pooled_v2_metrics": pooled_v2,
        "pooled_persistence_metrics": pooled_persistence,
        "pooled_skill_vs_persistence": pooled_skill,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("V2 EVALUATION COMPLETE")
    print("=" * 80)
    print(
        f"Forecast: {FORECAST_PATH}"
    )
    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
