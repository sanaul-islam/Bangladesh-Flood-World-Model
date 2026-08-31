from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset
from flood_world_model.models.world_model import FloodWorldModel
from flood_world_model.utils.paths import OUTPUT_DIR, V1_CHECKPOINT

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = V1_CHECKPOINT
OUTPUT_PATH = OUTPUT_DIR / "world_model_v1_test_metrics.json"


def masked_metrics(prediction, target, mask):
    valid = mask > 0.5

    prediction = prediction[valid]
    target = target[valid]

    if prediction.numel() == 0:
        return {
            "mae": None,
            "rmse": None,
            "bias": None,
            "correlation": None,
        }

    error = prediction - target

    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error ** 2))
    bias = torch.mean(error)

    if prediction.numel() > 1:
        correlation_matrix = torch.corrcoef(torch.stack([prediction, target]))
        correlation = float(correlation_matrix[0, 1])
    else:
        correlation = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "correlation": correlation,
    }


def main():
    print("=" * 80)
    print("WORLD MODEL V1 TEST EVALUATION")
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

    predictions = []
    targets = []
    masks = []

    print(f"Test samples: {len(dataset):,}")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            dynamic, static, target, target_mask = batch

            dynamic = dynamic.to(DEVICE)
            static = static.to(DEVICE)
            target = target[:, 0].to(DEVICE)
            target_mask = target_mask[:, 0].to(DEVICE)

            prediction = model(dynamic, static)

            river_mask = static[:, 3:4]
            mask = target_mask * river_mask

            predictions.append(prediction.cpu())
            targets.append(target.cpu())
            masks.append(mask.cpu())

            if batch_index == 1 or batch_index % 200 == 0:
                print(f"Evaluated {batch_index}/{len(loader)}")

    predictions = torch.cat(predictions, dim=0).flatten()
    targets = torch.cat(targets, dim=0).flatten()
    masks = torch.cat(masks, dim=0).flatten()

    metrics = masked_metrics(
        predictions,
        targets,
        masks,
    )

    print("=" * 80)
    print("V1 NORMALIZED TEST RESULTS")
    print("=" * 80)
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Bias: {metrics['bias']:.6f}")
    print(f"Correlation: {metrics['correlation']:.6f}")
    print(f"Valid target fraction: {float(masks.mean()):.6f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(CHECKPOINT_PATH),
                "version": "v1",
                "metrics": metrics,
                "valid_target_fraction": float(masks.mean()),
            },
            f,
            indent=2,
        )

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()