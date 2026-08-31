
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

from flood_world_model.datasets.v2_population import (
    V2PopulationDataset,
    build_normalization,
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
    / "data/features/training_v3/v2_population_normalization.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/world_model_v2_population_best.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/metrics/world_model_v2_population_training.json"
)

HISTORY_LENGTH = 14
HORIZON = 7

BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY = False

HIDDEN_CHANNELS = 16

EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

START_TEACHER_FORCING = 1.0
END_TEACHER_FORCING = 0.25

SEED = 42


def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=1.0,
    )

    mask = mask.float()

    denominator = (
        mask.sum()
        .clamp_min(1.0)
    )

    return (
        loss * mask
    ).sum() / denominator


def multi_horizon_loss(
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

    weight_total = torch.tensor(
        0.0,
        dtype=prediction.dtype,
        device=prediction.device,
    )

    for lead in range(
        prediction.size(1)
    ):
        lead_loss = masked_huber(
            prediction[:, lead],
            target[:, lead],
            mask[:, lead],
        )

        weight = lead_weights[
            lead
        ]

        total = (
            total
            + weight * lead_loss
        )

        weight_total = (
            weight_total
            + weight
        )

    return (
        total
        / weight_total
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    teacher_ratio: float,
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

        mask = batch[
            "mask"
        ].to(device)

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
                teacher_forcing_ratio=teacher_ratio,
            )

            loss = multi_horizon_loss(
                prediction,
                target,
                mask,
            )

            if training:
                loss.backward()

                clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP,
                )

                optimizer.step()

        total_loss += float(
            loss.detach().cpu()
        )

        batches += 1

    return (
        total_loss
        / max(batches, 1)
    )


def main() -> None:
    seed_everything(
        SEED
    )

    torch.set_num_threads(
        8
    )

    torch.set_num_interop_threads(
        1
    )

    device = torch.device(
        "cpu"
    )

    print("=" * 80)
    print("BANGLADESH FLOOD WORLD MODEL V2-POPULATION")
    print("7-DAY MULTI-HORIZON TRAINING")
    print("=" * 80)

    dynamic_ds = xr.open_zarr(
        DYNAMIC_PATH,
        consolidated=True,
    )

    static_ds = xr.open_zarr(
        STATIC_PATH,
        consolidated=True,
    )

    total_time = dynamic_ds.sizes[
        "time"
    ]

    train_end = int(
        total_time * 0.70
    )

    validation_end = int(
        total_time * 0.85
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
        f"Test: {validation_end} -> {total_time}"
    )

    print(
        "Checking population feature..."
    )

    population = (
        static_ds[
            "population_density"
        ]
        .values
        .astype(np.float32)
    )

    print(
        f"Population shape: {population.shape}"
    )

    print(
        f"Population min: {np.nanmin(population):.4f}"
    )

    print(
        f"Population max: {np.nanmax(population):.4f}"
    )

    print(
        f"Population mean: {np.nanmean(population):.4f}"
    )

    if population.shape != (
        60,
        45,
    ):
        dynamic_ds.close()
        static_ds.close()

        raise RuntimeError(
            f"Population must be 60x45, got {population.shape}"
        )

    print(
        "Creating training-only normalization..."
    )

    build_normalization(
        dynamic_ds=dynamic_ds,
        static_ds=static_ds,
        train_end=train_end,
        output_path=NORMALIZATION_PATH,
    )

    dynamic_ds.close()
    static_ds.close()

    print(
        f"Normalization: {NORMALIZATION_PATH}"
    )

    print(
        "Building training dataset..."
    )

    train_dataset = V2PopulationDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=0,
        end_index=train_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print(
        "Building validation dataset..."
    )

    validation_dataset = V2PopulationDataset(
        dynamic_path=DYNAMIC_PATH,
        static_path=STATIC_PATH,
        normalization_path=NORMALIZATION_PATH,
        start_index=train_end,
        end_index=validation_end,
        history_length=HISTORY_LENGTH,
        horizon=HORIZON,
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = FloodWorldModelV2(
        dynamic_channels=6,
        static_channels=12,
        hidden_channels=HIDDEN_CHANNELS,
        horizon=HORIZON,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

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

    best_validation_loss = float(
        "inf"
    )

    training_history = []

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
            teacher_ratio=ratio,
        )

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=None,
            device=device,
            teacher_ratio=0.0,
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "teacher_forcing_ratio": ratio,
            "learning_rate": current_lr,
        }

        training_history.append(
            record
        )

        print("-" * 80)
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: {train_loss:.8f}"
        )

        print(
            f"Validation loss: {validation_loss:.8f}"
        )

        print(
            f"Teacher forcing ratio: {ratio:.4f}"
        )

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "dynamic_channels": 6,
                "static_channels": 12,
                "hidden_channels": HIDDEN_CHANNELS,
                "horizon": HORIZON,
                "history_length": HISTORY_LENGTH,
                "task": "7_day_multi_horizon_discharge",
                "population_feature": True,
                "population_transform": "log1p",
                "population_source": (
                    "data/features/static_v3.zarr/population_density"
                ),
                "normalization_path": str(
                    NORMALIZATION_PATH
                ),
                "best_validation_loss": best_validation_loss,
                "train_end": train_end,
                "validation_end": validation_end,
                "total_time": total_time,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"Saved best checkpoint: {CHECKPOINT_PATH}"
            )

    results = {
        "model": "Bangladesh Flood World Model V2-Population",
        "task": "7-day multi-horizon discharge forecasting",
        "dynamic_channels": 6,
        "static_channels": 12,
        "population_feature": True,
        "population_transform": "log1p",
        "history_length": HISTORY_LENGTH,
        "horizon": HORIZON,
        "hidden_channels": HIDDEN_CHANNELS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_loss": best_validation_loss,
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "normalization": str(
            NORMALIZATION_PATH
        ),
        "training_history": training_history,
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

    print("=" * 80)
    print("V2-POPULATION TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best validation loss: {best_validation_loss:.8f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
