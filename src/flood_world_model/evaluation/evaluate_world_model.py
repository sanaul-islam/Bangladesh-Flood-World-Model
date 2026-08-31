from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset
from flood_world_model.models.world_model import FloodWorldModel
from flood_world_model.utils.paths import CHECKPOINT_DIR, NORMALIZATION_PATH, OUTPUT_DIR, V0_CHECKPOINT

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = V0_CHECKPOINT
OUTPUT_PATH = OUTPUT_DIR

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def masked_metrics(pred, target, mask):
    valid = mask > 0.5

    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "correlation": float("nan"),
        }

    error = pred - target

    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error ** 2))
    bias = torch.mean(error)

    if pred.numel() > 1:
        correlation = torch.corrcoef(torch.stack([pred, target]))[0, 1]
        correlation = float(correlation)
    else:
        correlation = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "correlation": correlation,
    }


def main():
    print("=" * 80)
    print("WORLD MODEL V0 FINAL EVALUATION")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model = FloodWorldModel(
        dynamic_channels=checkpoint["dynamic_channels"],
        static_channels=checkpoint["static_channels"],
        hidden_channels=checkpoint["hidden_channels"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = FloodWorldModelDataset(
        split="test",
        history_days=14,
        forecast_days=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
        normalization = json.load(f)

    discharge_stats = normalization["river_discharge"]
    discharge_mean = float(discharge_stats["mean"])
    discharge_std = max(float(discharge_stats["std"]), 1e-8)

    all_model_pred = []
    all_target = []
    all_persistence = []
    all_mask = []

    print(f"Test samples: {len(dataset):,}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            dynamic, static, target, target_mask = batch

            dynamic = dynamic.to(DEVICE)
            static = static.to(DEVICE)
            target = target[:, 0].to(DEVICE)
            target_mask = target_mask[:, 0].to(DEVICE)

            river_mask = static[:, 3:4]
            mask = target_mask * river_mask

            prediction = model(dynamic, static)

            persistence = dynamic[:, -1, 5:6]

            all_model_pred.append(prediction.cpu())
            all_target.append(target.cpu())
            all_persistence.append(persistence.cpu())
            all_mask.append(mask.cpu())

            if batch_idx == 1 or batch_idx % 100 == 0:
                print(f"Evaluated {batch_idx}/{len(loader)}")

    model_pred = torch.cat(all_model_pred, dim=0).flatten()
    target = torch.cat(all_target, dim=0).flatten()
    persistence = torch.cat(all_persistence, dim=0).flatten()
    mask = torch.cat(all_mask, dim=0).flatten()

    model_metrics = masked_metrics(
        model_pred,
        target,
        mask,
    )

    persistence_metrics = masked_metrics(
        persistence,
        target,
        mask,
    )

    model_physical = model_pred * discharge_std + discharge_mean
    target_physical = target * discharge_std + discharge_mean
    persistence_physical = persistence * discharge_std + discharge_mean

    physical_model = masked_metrics(
        model_physical,
        target_physical,
        mask,
    )

    physical_persistence = masked_metrics(
        persistence_physical,
        target_physical,
        mask,
    )

    model_rmse = model_metrics["rmse"]
    persistence_rmse = persistence_metrics["rmse"]

    skill = 1.0 - (
        model_rmse / max(persistence_rmse, 1e-12)
    )

    print("\n" + "=" * 80)
    print("NORMALIZED RESULTS")
    print("=" * 80)

    print(f"World model MAE: {model_metrics['mae']:.6f}")
    print(f"Persistence MAE: {persistence_metrics['mae']:.6f}")
    print(f"World model RMSE: {model_metrics['rmse']:.6f}")
    print(f"Persistence RMSE: {persistence_metrics['rmse']:.6f}")
    print(f"World model bias: {model_metrics['bias']:.6f}")
    print(f"Persistence bias: {persistence_metrics['bias']:.6f}")
    print(f"World model correlation: {model_metrics['correlation']:.6f}")
    print(f"Persistence correlation: {persistence_metrics['correlation']:.6f}")
    print(f"Skill vs persistence: {skill:.2%}")

    print("\n" + "=" * 80)
    print("PHYSICAL DISCHARGE RESULTS")
    print("=" * 80)

    print(f"World model MAE: {physical_model['mae']:.4f} m3/s")
    print(f"Persistence MAE: {physical_persistence['mae']:.4f} m3/s")
    print(f"World model RMSE: {physical_model['rmse']:.4f} m3/s")
    print(f"Persistence RMSE: {physical_persistence['rmse']:.4f} m3/s")
    print(f"World model bias: {physical_model['bias']:.4f} m3/s")
    print(f"Persistence bias: {physical_persistence['bias']:.4f} m3/s")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if model_rmse < persistence_rmse:
        print("✅ World model beats persistence.")
    else:
        print("❌ World model does not beat persistence.")

    metrics = {
        "checkpoint": str(CHECKPOINT_PATH),
        "best_validation_loss": checkpoint["best_val_loss"],
        "normalized": {
            "model": model_metrics,
            "persistence": persistence_metrics,
            "skill_vs_persistence": skill,
        },
        "physical_units_m3_s": {
            "model": physical_model,
            "persistence": physical_persistence,
        },
    }

    with open(
        OUTPUT_PATH / "world_model_v0_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics: {OUTPUT_PATH / 'world_model_v0_test_metrics.json'}")


if __name__ == "__main__":
    main()