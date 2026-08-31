from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.xz = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.hz = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

        self.xr = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.hr = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

        self.xh = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.hh = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)

        z = torch.sigmoid(self.xz(x) + self.hz(h))
        r = torch.sigmoid(self.xr(x) + self.hr(h))
        candidate = torch.tanh(self.xh(x) + self.hh(r * h))

        return (1.0 - z) * h + z * candidate


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FloodWorldModel(nn.Module):
    def __init__(self, dynamic_channels: int, static_channels: int, hidden_channels: int = 32):
        super().__init__()

        self.dynamic_encoder = SpatialEncoder(dynamic_channels, hidden_channels)
        self.static_encoder = StaticEncoder(static_channels, hidden_channels)

        self.static_projection = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.gru = ConvGRUCell(hidden_channels * 2, hidden_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        static_features = self.static_encoder(static)
        static_features = self.static_projection(static_features)

        h = None

        for t in range(dynamic.size(1)):
            step = dynamic[:, t]
            dynamic_features = self.dynamic_encoder(step)
            fused = torch.cat([dynamic_features, static_features], dim=1)
            h = self.gru(fused, h)

        h = self.fusion(torch.cat([h, static_features], dim=1))
        return self.head(h)