
from __future__ import annotations

import json
import time

import torch
from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset
from flood_world_model.models.world_model import FloodWorldModel
from flood_world_model.training.losses import masked_huber_loss, masked_mae, masked_rmse
from flood_world_model.utils.paths import CHECKPOINT_DIR

DEVICE = torch.device("cpu")

EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 16

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = CHECKPOINT_DIR / "world_model_v0_best.pt"


def create_loader(split: str, shuffle: bool):
    dataset = FloodWorldModelDataset(
        split=split,
        history_days=14,
        forecast_days=1,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def run_epoch(model, loader, optimizer=None, epoch=0, total_epochs=1):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        dynamic, static, target, target_mask = batch

        dynamic = dynamic.to(DEVICE)
        static = static.to(DEVICE)
        target = target[:, 0].to(DEVICE)
        target_mask = target_mask[:, 0].to(DEVICE)

        river_mask = static[:, 3:4]
        loss_mask = target_mask * river_mask

        if training:
            optimizer.zero_grad(set_to_none=True)

        prediction = model(dynamic, static)

        loss = masked_huber_loss(
            prediction,
            target,
            loss_mask,
        )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

        mae = masked_mae(
            prediction.detach(),
            target,
            loss_mask,
        )

        rmse = masked_rmse(
            prediction.detach(),
            target,
            loss_mask,
        )

        total_loss += float(loss.detach())
        total_mae += float(mae.detach())
        total_rmse += float(rmse.detach())

        if batch_idx == 1 or batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{total_epochs} {'train' if training else 'val'} batch {batch_idx}/{len(loader)} loss={float(loss):.6f} elapsed={elapsed:.1f}s")

    count = max(len(loader), 1)

    return (
        total_loss / count,
        total_mae / count,
        total_rmse / count,
    )


def main():
    print("=" * 80)
    print("TRAINING FLOOD WORLD MODEL V0")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden channels: {HIDDEN_CHANNELS}")

    print("\nBuilding training loader...")
    train_loader = create_loader("train", True)

    print("\nBuilding validation loader...")
    val_loader = create_loader("val", False)

    model = FloodWorldModel(
        dynamic_channels=6,
        static_channels=11,
        hidden_channels=HIDDEN_CHANNELS,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTrainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")

    history = []

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 80)

        epoch_start = time.time()

        train_loss, train_mae, train_rmse = run_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            EPOCHS,
        )

        with torch.no_grad():
            val_loss, val_mae, val_rmse = run_epoch(
                model,
                val_loader,
                None,
                epoch,
                EPOCHS,
            )

        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch} complete in {elapsed:.1f}s")
        print(f"Train loss: {train_loss:.6f}")
        print(f"Val loss: {val_loss:.6f}")
        print(f"Train MAE: {train_mae:.6f}")
        print(f"Val MAE: {val_mae:.6f}")
        print(f"Train RMSE: {train_rmse:.6f}")
        print(f"Val RMSE: {val_rmse:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dynamic_channels": 6,
                    "static_channels": 11,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "history_days": 14,
                    "forecast_days": 1,
                    "best_val_loss": best_val_loss,
                },
                BEST_PATH,
            )

            print(f"✅ Best model saved: {BEST_PATH}")

    with open(
        CHECKPOINT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {BEST_PATH}")


if __name__ == "__main__":
    main()
