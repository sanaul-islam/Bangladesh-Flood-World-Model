from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flood_world_model.models.world_model import FloodWorldModel


class V0Predictor:
    def __init__(self, project_root: Path, device: str = "cpu"):
        self.project_root = Path(project_root)
        self.device = torch.device(device)

        self.checkpoint_path = self.project_root / "models/checkpoints/world_model_v0_best.pt"
        self.normalization_path = self.project_root / "data/features/training_v3/normalization.json"

        with open(self.normalization_path, "r", encoding="utf-8") as f:
            self.normalization = json.load(f)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model = FloodWorldModel(
            dynamic_channels=checkpoint["dynamic_channels"],
            static_channels=checkpoint["static_channels"],
            hidden_channels=checkpoint["hidden_channels"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        stats = self.normalization["river_discharge"]
        self.discharge_mean = float(stats["mean"])
        self.discharge_std = max(float(stats["std"]), 1e-8)

    def normalize_dynamic(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = self.normalization[name]

        if stats["type"] == "binary":
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(values, 0.0, 1.0).astype(np.float32)

        mean = float(stats["mean"])
        std = max(float(stats["std"]), 1e-8)

        values = (values - mean) / std
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        return values.astype(np.float32)

    def normalize_static(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = self.normalization[f"static_{name}"]

        if stats["type"] in {"binary", "categorical"}:
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        mean = float(stats["mean"])
        std = max(float(stats["std"]), 1e-8)

        values = (values - mean) / std
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        return values.astype(np.float32)

    def predict(self, dynamic_array: np.ndarray, static_array: np.ndarray) -> np.ndarray:
        if dynamic_array.ndim != 4:
            raise ValueError(f"Expected dynamic shape [T,C,H,W], got {dynamic_array.shape}")

        if static_array.ndim != 3:
            raise ValueError(f"Expected static shape [C,H,W], got {static_array.shape}")

        dynamic_tensor = torch.from_numpy(dynamic_array[None]).to(self.device)
        static_tensor = torch.from_numpy(static_array[None]).to(self.device)

        with torch.no_grad():
            prediction = self.model(dynamic_tensor, static_tensor)

        prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()

        return prediction.astype(np.float32)

    def denormalize_discharge(self, values: np.ndarray) -> np.ndarray:
        return values * self.discharge_std + self.discharge_mean