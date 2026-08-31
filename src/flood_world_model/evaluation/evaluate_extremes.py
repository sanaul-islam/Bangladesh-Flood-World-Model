from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset
from flood_world_model.models.world_model import FloodWorldModel
from flood_world_model.utils.paths import OUTPUT_DIR, V0_CHECKPOINT

DEVICE = torch.device("cpu")

CHECKPOINT = V0_CHECKPOINT


def metrics(pred, target, mask):
    valid = mask > 0.5

    pred = pred[valid]
    target = target[valid]

    error = pred - target

    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error ** 2))
    bias = torch.mean(error)

    return float(mae), float(rmse), float(bias)


def main():
    print("=" * 80)
    print("EXTREME DISCHARGE EVALUATION")
    print("=" * 80)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

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

    predictions = []
    targets = []
    masks = []

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            dynamic, static, target, target_mask = batch

            dynamic = dynamic.to(DEVICE)
            static = static.to(DEVICE)
            target = target[:, 0].to(DEVICE)
            target_mask = target_mask[:, 0].to(DEVICE)

            river_mask = static[:, 3:4]
            mask = target_mask * river_mask

            prediction = model(dynamic, static)

            predictions.append(prediction.cpu())
            targets.append(target.cpu())
            masks.append(mask.cpu())

            if i == 1 or i % 100 == 0:
                print(f"Evaluated {i}/{len(loader)}")

    pred = torch.cat(predictions).flatten()
    target = torch.cat(targets).flatten()
    mask = torch.cat(masks).flatten()

    valid = mask > 0.5

    target_valid = target[valid]

    q90 = torch.quantile(target_valid, 0.90)
    q95 = torch.quantile(target_valid, 0.95)
    q99 = torch.quantile(target_valid, 0.99)

    thresholds = {
        "all": -float("inf"),
        "top_10_percent": float(q90),
        "top_5_percent": float(q95),
        "top_1_percent": float(q99),
    }

    results = {}

    for name, threshold in thresholds.items():
        subset = valid & (target >= threshold)

        mae, rmse, bias = metrics(
            pred,
            target,
            subset.float(),
        )

        results[name] = {
            "count": int(subset.sum()),
            "mae_normalized": mae,
            "rmse_normalized": rmse,
            "bias_normalized": bias,
        }

        print(f"{name}: count={int(subset.sum())} mae={mae:.6f} rmse={rmse:.6f} bias={bias:.6f}")

    print("\n" + "=" * 80)
    print("EXTREME-EVENT RESULTS")
    print("=" * 80)

    print(f"90th percentile threshold: {float(q90):.6f}")
    print(f"95th percentile threshold: {float(q95):.6f}")
    print(f"99th percentile threshold: {float(q99):.6f}")

    output_file = OUTPUT_DIR / "world_model_v0_extreme_metrics.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()