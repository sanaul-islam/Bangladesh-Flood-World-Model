from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flood_world_model.datasets.multihorizon import (
    MultiHorizonFloodDataset,
    build_training_normalization,
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
    / "outputs/metrics/world_model_v2_training.json"
)

HISTORY_LENGTH = 14
HORIZON = 7

BATCH_SIZE = 2
NUM_WORKERS = 0

HIDDEN_CHANNELS = 16

EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

START_TEACHER_FORCING = 1.0
END_TEACHER_FORCING = 0.25

SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_split_indices(
    time_values: np.ndarray,
) -> tuple[int, int, int]:
    """
    Chronological 70/15/15 split.

    Replace these dates with the exact V0 split boundaries
    if your original V0 experiment used different boundaries.
    """

    n = len(time_values)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    return (
        0,
        train_end,
        val_end,
    )


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    raw = F.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=1.0,
    )

    weights = mask.float()

    denominator = (
        weights.sum()
        .clamp_min(1.0)
    )

    return (
        raw * weights
    ).sum() / denominator


def weighted_multi_horizon_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    lead_weights = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.1,
            1.2,
            1.3,
        ],
        dtype=prediction.dtype,
        device=prediction.device,
    )

    total = torch.tensor(
        0.0,
        dtype=prediction.dtype,
        device=prediction.device,
    )

    weight_sum = torch.tensor(
        0.0,
        dtype=prediction.dtype,
        device=prediction.device,
    )

    for lead in range(
        prediction.size(1)
    ):
        loss = masked_huber_loss(
            prediction[:, lead],
            target[:, lead],
            mask[:, lead],
        )

        weight = lead_weights[
            lead
        ]

        total = (
            total
            + weight * loss
        )

        weight_sum = (
            weight_sum
            + weight
        )

    return total / weight_sum


def teacher_forcing_ratio(
    epoch: int,
) -> float:
    if EPOCHS <= 1:
        return END_TEACHER_FORCING

    progress = epoch / float(
        EPOCHS - 1
    )

    ratio = (
        START_TEACHER_FORCING
        + progress
        * (
            END_TEACHER_FORCING
            - START_TEACHER_FORCING
        )
    )

    return float(
        np.clip(
            ratio,
            END_TEACHER_FORCING,
            START_TEACHER_FORCING,
        )
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    forcing_ratio: float,
) -> float:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0

    for batch in loader:
        history = batch[
            "history"
        ].to(
            device,
            non_blocking=False,
        )

        static = batch[
            "static"
        ].to(
            device,
            non_blocking=False,
        )

        future_forcing = batch[
            "future_forcing"
        ].to(
            device,
            non_blocking=False,
        )

        initial_discharge = batch[
            "initial_discharge"
        ].to(
            device,
            non_blocking=False,
        )

        target = batch[
            "target"
        ].to(
            device,
            non_blocking=False,
        )

        mask = batch[
            "mask"
        ].to(
            device,
            non_blocking=False,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            prediction = model(
                history=history,
                static=static,
                future_forcing=future_forcing,
                initial_discharge=initial_discharge,
                target_discharge=target,
                teacher_forcing_ratio=forcing_ratio,
            )

            loss = weighted_multi_horizon_loss(
                prediction,
                target,
                mask,
            )

            if training:
                loss.backward()

                clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

        total_loss += float(
            loss.detach().cpu()
        )

        batches += 1

    return total_loss / max(
        batches,
        1,
    )


def main() -> None:
    seed_everything(SEED)

    device = torch.device(
        "cpu"
    )

    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL V2")
    print("7-DAY MULTI-HORIZON TRAINING")
    print("=" * 80)

    if not DYNAMIC_PATH.exists():
        raise FileNotFoundError(
            DYNAMIC_PATH
        )

    if not STATIC_PATH.exists():
        raise FileNotFoundError(
            STATIC_PATH
        )

    print("Loading dataset metadata...")

    dynamic_ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    time_values = (
        dynamic_ds.time.values
    )

    total_time = len(
        time_values
    )

    train_start, train_end, val_end = (
        get_split_indices(
            time_values
        )
    )

    dynamic_ds.close()

    print(f"Total time steps: {total_time}")
    print(f"Train indices: {train_start} -> {train_end}")
    print(f"Validation indices: {train_end} -> {val_end}")
    print(f"Test indices: {val_end} -> {total_time}")

    print("Creating training-only normalization...")

    dynamic_ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    static_ds = xr.open_zarr(
        STATIC_PATH,
        consolidated=True,
    )

    normalization = (
        build_training_normalization(
            dynamic_ds=dynamic_ds,
            static_ds=static_ds,
            train_end_index=train_end,
            output_path=NORMALIZATION_PATH,
        )
    )

    dynamic_ds.close()
    static_ds.close()

    print(
        f"Normalization: {NORMALIZATION_PATH}"
    )

    print("Building training dataset...")

    train_dataset = MultiHorizonFloodDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=train_start,
        end_index=train_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print("Building validation dataset...")

    val_dataset = MultiHorizonFloodDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=train_end,
        end_index=val_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = FloodWorldModelV2(
        dynamic_channels=6,
        static_channels=11,
        hidden_channels=HIDDEN_CHANNELS,
        horizon=HORIZON,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_loss = float(
        "inf"
    )

    history = []

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        ratio = teacher_forcing_ratio(
            epoch - 1
        )

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            forcing_ratio=ratio,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            forcing_ratio=0.0,
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "teacher_forcing_ratio": ratio,
            "learning_rate": current_lr,
        }

        history.append(
            record
        )

        print("-" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Validation loss: {val_loss:.8f}")
        print(f"Teacher forcing ratio: {ratio:.4f}")
        print(f"Learning rate: {current_lr:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "dynamic_channels": 6,
                "static_channels": 11,
                "hidden_channels": HIDDEN_CHANNELS,
                "horizon": HORIZON,
                "history_length": HISTORY_LENGTH,
                "checkpoint_type": "world_model_v2_multihorizon",
                "best_validation_loss": best_val_loss,
                "normalization_path": str(
                    NORMALIZATION_PATH
                ),
                "train_start_index": train_start,
                "train_end_index": train_end,
                "validation_end_index": val_end,
                "total_time": total_time,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"Saved best checkpoint: {CHECKPOINT_PATH}"
            )

    metrics = {
        "model": "World Model V2",
        "task": "7-day multi-horizon discharge forecasting",
        "history_length": HISTORY_LENGTH,
        "horizon": HORIZON,
        "batch_size": BATCH_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "start_teacher_forcing": START_TEACHER_FORCING,
        "end_teacher_forcing": END_TEACHER_FORCING,
        "best_validation_loss": best_val_loss,
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "normalization": str(
            NORMALIZATION_PATH
        ),
        "history": history,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("=" * 80)
    print("V2 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()