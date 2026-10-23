"""Encoder interface and the classifier built on top of it.

Every visual backbone (the CNN trained from scratch, a pretrained timm model)
is an :class:`Encoder`: image in, feature vector out. The classification head
is a separate module, so the same encoder can later feed the colorization
decoder and the caption decoder in Milestone 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module, ABC):
    """Image -> feature vector of length ``out_dim``."""

    out_dim: int

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, H, W)`` -> last feature map ``(B, out_dim, h, w)``.

        Grad-CAM reads its gradients from this map, and image-to-image models
        decode from it.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, H, W)`` -> ``(B, out_dim)`` by global average pooling."""
        return torch.flatten(F.adaptive_avg_pool2d(self.forward_features(x), 1), 1)


class Classifier(nn.Module):
    """Encoder + dropout + linear layer -> class logits."""

    def __init__(self, encoder: Encoder, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.out_dim, num_classes))
        self.encoder_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def freeze_encoder(self) -> None:
        """Train only the head (a "linear probe" on a pretrained encoder)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True

    def train(self, mode: bool = True) -> "Classifier":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()   # a frozen encoder keeps its BatchNorm statistics fixed
        return self
