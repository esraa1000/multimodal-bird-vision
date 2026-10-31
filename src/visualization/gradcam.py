"""Class-specific Grad-CAM for a classification encoder's final spatial features.

Selvaraju et al., ICCV 2017: https://arxiv.org/abs/1610.02391
Use raw class logits, spatially averaged gradients, and a ReLU-weighted sum.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def compute_gradcam(model: nn.Module, images: torch.Tensor,
                    class_id: int | None = None) -> dict:
    """Explain one image without updating parameters, buffers, or parameter grads.

    The model must already be in evaluation mode. Making the input require
    gradients also supports completely frozen encoders. The returned heatmap
    is CPU float32, normalized to [0, 1], and aligned with the input tensor.
    An all-zero map means no positive evidence at the chosen layer, not an error.
    """
    if model.training:
        raise ValueError("Grad-CAM requires model.eval() to keep BatchNorm statistics fixed")
    if images.ndim != 4 or images.shape[0] != 1 or not images.is_floating_point():
        raise ValueError("Grad-CAM expects one floating-point image with shape (1, C, H, W)")
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "feature_map"):
        name, layer = "feature_map", encoder.feature_map
    else:
        layers = [(name, layer) for name, layer in encoder.named_modules()
                  if isinstance(layer, nn.Conv2d)]
        if not layers:
            raise ValueError("Grad-CAM needs a convolutional encoder")
        name, layer = layers[-1]
    layer_name = f"encoder{'.' + name if name else ''}" if encoder is not model else name
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(images.detach().clone().requires_grad_(True))
            target = int(logits[0].argmax()) if class_id is None else class_id
            if not 0 <= target < logits.shape[1]:
                raise ValueError(f"Grad-CAM class id must be between 0 and {logits.shape[1] - 1}")
            activation = captured[-1]
            gradient, = torch.autograd.grad(logits[0, target], activation)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            cam = cam / cam.max().clamp_min(torch.finfo(cam.dtype).tiny)
        return {"heatmap": cam.detach().float().cpu(), "logits": logits.detach(),
                "class_id": target, "layer": layer_name,
                "feature_map_size": list(activation.shape[-2:])}
    finally:
        handle.remove()
