from __future__ import annotations

import torch
import torch.nn.functional as F


def flood_weighted_huber_loss(prediction, target, mask, high_flow_threshold=1.0, alpha=3.0):
    base_loss = F.huber_loss(prediction, target, reduction="none")
    severity = torch.clamp((target - high_flow_threshold) / 3.0, min=0.0)
    weights = 1.0 + alpha * severity
    weighted_loss = base_loss * weights * mask
    denominator = (weights * mask).sum().clamp_min(1.0)
    return weighted_loss.sum() / denominator


def masked_mae(prediction, target, mask):
    error = torch.abs(prediction - target) * mask
    return error.sum() / mask.sum().clamp_min(1.0)


def masked_rmse(prediction, target, mask):
    error = ((prediction - target) ** 2) * mask
    return torch.sqrt(error.sum() / mask.sum().clamp_min(1.0))