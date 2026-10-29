
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
        trainable_stages: list[str] | None = None,
        freeze_bn_stats: bool = False,
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
        self.trainable_stages = trainable_stages
        self.freeze_bn_stats = freeze_bn_stats
        if trainable_stages is not None:
            if (not isinstance(trainable_stages, list) or not trainable_stages
                    or any(not isinstance(name, str) or not name for name in trainable_stages)):
                raise ValueError("trainable_stages must be a non-empty list of module names")
            for name in trainable_stages:
                try:
                    stage = self.model.get_submodule(name)
                except AttributeError as exc:
                    raise ValueError(f"Unknown stage '{name}' for {backbone}") from exc
                if not any(True for _ in stage.parameters()):
                    raise ValueError(f"Stage '{name}' has no parameters to fine-tune")
            self.model.requires_grad_(False)
            for name in trainable_stages:
                self.model.get_submodule(name).requires_grad_(True)

    def train(self, mode: bool = True) -> "TimmEncoder":
        super().train(mode)
        if self.trainable_stages is not None:
            # Freeze BatchNorm statistics and dropout in all earlier stages.
            self.model.eval()
            if mode:
                for name in self.trainable_stages:
                    self.model.get_submodule(name).train()
        if self.freeze_bn_stats:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Evaluation mode fixes running statistics without disabling
                    # gradients for trainable BatchNorm scale/bias parameters.
                    module.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)
