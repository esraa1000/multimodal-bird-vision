"""CNN encoder trained from scratch: the baseline the project requires.

Layout for the default ``depth=4, width=64`` on a 224x224 image::

    stem     conv3x3 stride 2                     ->  64 x 112 x 112
    stage 1  blocks_per_stage x block             ->  64 x 112 x 112
    stage 2  maxpool, blocks_per_stage x block    -> 128 x  56 x  56
    stage 3  maxpool, blocks_per_stage x block    -> 256 x  28 x  28
    stage 4  maxpool, blocks_per_stage x block    -> 512 x  14 x  14   <- forward_features
    global average pool                           -> 512-d vector      <- forward

Channels double whenever the resolution halves, so every stage costs about
the same compute (~2.8 G multiply-adds in total, 4.7M parameters with plain blocks).

* ``residual=False``: a block is conv3x3 -> BatchNorm -> ReLU (VGG style).
* ``residual=True``: a block is two such convs with a skip connection
  (ResNet "basic block"), so it holds twice as many convs. For a fair
  plain-vs-residual comparison use ``blocks_per_stage`` 4 (plain) vs 2 (residual).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import Encoder
from src.registry import MODELS


def conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),  # BN has its own bias
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    """``relu(conv-bn-relu-conv-bn(x) + shortcut(x))``."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn_relu(c_in, c_out),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        # 1x1 conv when the channel count changes, so the shapes can be added
        self.shortcut = nn.Identity() if c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


@MODELS.register("scratch_cnn")
class ScratchCNN(Encoder):
    def __init__(self, depth: int = 4, width: int = 64, blocks_per_stage: int = 2,
                 residual: bool = False, stem_stride: int = 2, max_width: int = 512,
                 in_channels: int = 3):
        """
        Args:
            depth: number of stages; each stage after the first halves the resolution.
            width: channels of the first stage (doubled per stage, capped at ``max_width``).
            blocks_per_stage: blocks in each stage (see the module docstring).
            residual: use residual blocks instead of plain conv-BN-ReLU.
            stem_stride: stride of the first conv (2 halves the cost of everything after it).
            in_channels: 3 for RGB; 1 for a grayscale input (colorization, Milestone 2).
        """
        super().__init__()
        widths = [min(width * 2 ** i, max_width) for i in range(depth)]
        self.stem = conv_bn_relu(in_channels, widths[0], stride=stem_stride)

        stages, c_in = [], widths[0]
        for i, c_out in enumerate(widths):
            layers: list[nn.Module] = [nn.MaxPool2d(2)] if i > 0 else []
            for _ in range(blocks_per_stage):
                layers.append(ResidualBlock(c_in, c_out) if residual else conv_bn_relu(c_in, c_out))
                c_in = c_out
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.out_dim = widths[-1]
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Feature map after every stage (high to low resolution), e.g. for U-Net skips."""
        x = self.stem(x)
        maps = []
        for stage in self.stages:
            x = stage(x)
            maps.append(x)
        return maps

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x)[-1]
