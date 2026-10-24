"""Small CNN baseline for log-Mel spectrograms."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CNNBaseline(nn.Module):
    """Four-block convolutional binary classifier with one synthetic logit."""

    def __init__(self) -> None:
        super().__init__()
        channels = (1, 32, 64, 96, 128)
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("CNNBaseline expects [batch, 1, mels, frames]")
        x = self.features(features)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
