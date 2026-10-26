
"""VGG-style CNN encoder trained from scratch."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import Encoder
from src.registry import MODELS


def vgg_block(c_in: int, c_out: int, num_convs: int) -> nn.Sequential:
    """Build a block containing repeated Conv-BN-ReLU layers."""
    layers = []

    for i in range(num_convs):
        layers.extend([
            nn.Conv2d(
                c_in if i == 0 else c_out,
                c_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ])

    return nn.Sequential(*layers)


@MODELS.register("vgg_cnn")
class VGGCNN(Encoder):
    def __init__(
        self,
        width: int = 32,
        in_channels: int = 3,
        num_convs: int = 2,
    ):
        super().__init__()

        channels = [width, width * 2, width * 4, width * 8]
        stages = []
        c_in = in_channels

        for c_out in channels:
            stages.append(
                nn.Sequential(
                    vgg_block(c_in, c_out, num_convs),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            c_in = c_out

        self.features = nn.Sequential(*stages)
        self.out_dim = channels[-1]

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
