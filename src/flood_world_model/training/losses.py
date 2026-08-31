from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none")
    loss = loss * mask
    denominator = mask.sum().clamp_min(1.0)
    return loss.sum() / denominator


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = torch.abs(pred - target) * mask
    return error.sum() / mask.sum().clamp_min(1.0)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = ((pred - target) ** 2) * mask
    return torch.sqrt(error.sum() / mask.sum().clamp_min(1.0))