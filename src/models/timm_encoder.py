
"""Pretrained image encoders for transfer learning."""

from __future__ import annotations

import torch
from torch import nn
import timm

from src.models.base import Encoder
from src.registry import MODELS


@MODELS.register("timm_encoder")
class TimmEncoder(Encoder):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()

        self.backbone_name = backbone

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

        self.out_dim = self.model.num_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)
