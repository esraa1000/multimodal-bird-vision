
"""Classifier with a configurable nonlinear head for transfer learning."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import Encoder


class TransferClassifier(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int = 200,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.encoder_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        self.encoder_frozen = False

    def train(self, mode: bool = True) -> "TransferClassifier":
        super().train(mode)

        if self.encoder_frozen:
            self.encoder.eval()

        return self
