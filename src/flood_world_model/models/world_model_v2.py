from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None,
    ) -> torch.Tensor:
        if h is None:
            h = torch.zeros(
                x.size(0),
                self.hidden_channels,
                x.size(2),
                x.size(3),
                dtype=x.dtype,
                device=x.device,
            )

        combined = torch.cat(
            [x, h],
            dim=1,
        )

        gates = self.gates(
            combined
        )

        update_gate, reset_gate = torch.chunk(
            gates,
            2,
            dim=1,
        )

        update_gate = torch.sigmoid(
            update_gate
        )

        reset_gate = torch.sigmoid(
            reset_gate
        )

        candidate_input = torch.cat(
            [
                x,
                reset_gate * h,
            ],
            dim=1,
        )

        candidate = torch.tanh(
            self.candidate(
                candidate_input
            )
        )

        h_next = (
            (1.0 - update_gate) * h
            + update_gate * candidate
        )

        return h_next


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class FloodWorldModelV2(nn.Module):
    """
    Seven-day multi-horizon hydrological world model.

    Encoder:
        14 days of historical dynamic state.

    Decoder:
        7 future days.

    Decoder input for each future day:
        future rainfall-derived forcing + previous discharge.

    During training:
        scheduled sampling mixes ground-truth and predicted discharge.

    During inference:
        previous predicted discharge is always fed back.
    """

    def __init__(
        self,
        dynamic_channels: int = 6,
        static_channels: int = 11,
        hidden_channels: int = 16,
        horizon: int = 7,
    ) -> None:
        super().__init__()

        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.hidden_channels = hidden_channels
        self.horizon = horizon

        self.dynamic_encoder = SpatialEncoder(
            dynamic_channels,
            hidden_channels,
        )

        self.static_encoder = SpatialEncoder(
            static_channels,
            hidden_channels,
        )

        self.static_projection = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.encoder_gru = ConvGRUCell(
            hidden_channels,
            hidden_channels,
        )

        self.decoder_encoder = SpatialEncoder(
            dynamic_channels,
            hidden_channels,
        )

        self.decoder_gru = ConvGRUCell(
            hidden_channels,
            hidden_channels,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
            ),
        )

    def encode(
        self,
        history: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        static_features = self.static_encoder(
            static
        )

        static_features = self.static_projection(
            static_features
        )

        hidden = None

        history_length = history.size(1)

        for t in range(history_length):
            dynamic_features = self.dynamic_encoder(
                history[:, t]
            )

            combined = (
                dynamic_features
                + static_features
            )

            hidden = self.encoder_gru(
                combined,
                hidden,
            )

        return hidden, static_features

    def decode(
        self,
        hidden: torch.Tensor,
        static_features: torch.Tensor,
        future_forcing: torch.Tensor,
        previous_discharge: torch.Tensor,
        target_discharge: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        predictions = []

        decoder_hidden = hidden

        for lead in range(self.horizon):
            forcing = future_forcing[:, lead]

            previous = previous_discharge.unsqueeze(
                1
            )

            decoder_input = torch.cat(
                [
                    forcing,
                    previous,
                ],
                dim=1,
            )

            decoder_features = self.decoder_encoder(
                decoder_input
            )

            decoder_features = (
                decoder_features
                + static_features
            )

            decoder_hidden = self.decoder_gru(
                decoder_features,
                decoder_hidden,
            )

            fused = self.fusion(
                torch.cat(
                    [
                        decoder_hidden,
                        static_features,
                    ],
                    dim=1,
                )
            )

            prediction = self.output_head(
                fused
            )

            predictions.append(
                prediction
            )

            predicted_previous = (
                prediction[:, 0]
            )

            if (
                self.training
                and target_discharge is not None
                and lead < self.horizon - 1
            ):
                use_teacher = (
                    torch.rand(
                        prediction.size(0),
                        device=prediction.device,
                    )
                    < teacher_forcing_ratio
                )

                true_previous = (
                    target_discharge[
                        :,
                        lead,
                    ]
                )

                previous_discharge = torch.where(
                    use_teacher[:, None, None],
                    true_previous,
                    predicted_previous,
                )

            else:
                previous_discharge = (
                    predicted_previous
                )

        return torch.stack(
            predictions,
            dim=1,
        )

    def forward(
        self,
        history: torch.Tensor,
        static: torch.Tensor,
        future_forcing: torch.Tensor,
        initial_discharge: torch.Tensor,
        target_discharge: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        history:
            [B, 14, 6, H, W]

        static:
            [B, 11, H, W]

        future_forcing:
            [B, 7, 5, H, W]

        initial_discharge:
            [B, H, W]

        target_discharge:
            [B, 7, H, W]

        returns:
            [B, 7, H, W]
        """

        hidden, static_features = self.encode(
            history,
            static,
        )

        predictions = self.decode(
            hidden=hidden,
            static_features=static_features,
            future_forcing=future_forcing,
            previous_discharge=initial_discharge,
            target_discharge=target_discharge,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return predictions[:, :, 0]