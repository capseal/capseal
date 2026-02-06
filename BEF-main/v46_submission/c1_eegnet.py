from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetC1(nn.Module):
    """Compact EEGNet architecture adapted for reaction-time regression."""

    def __init__(
        self,
        n_chans: int = 129,
        n_samples: int = 200,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_samples = n_samples

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(F1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (self.n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        final_samples = n_samples // 4 // 8
        self.flatten = nn.Flatten()
        self.regression_head = nn.Linear(F2 * final_samples, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return self.regression_head(x)
