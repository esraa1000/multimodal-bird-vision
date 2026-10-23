"""Optimizer and learning-rate schedule built from the ``train`` config section.

    train:
      optimizer: adamw        # adamw | adam | sgd
      lr: 1.0e-3
      weight_decay: 0.05
      scheduler: cosine       # cosine | none
      warmup_epochs: 2        # linear warm-up from 10% of lr, then the schedule
      min_lr: 0.0             # floor of the cosine schedule
"""

from __future__ import annotations

import torch
from torch.optim import lr_scheduler


def build_optimizer(model: torch.nn.Module, train_cfg: dict,
                    backbone_lr_mult: float = 1.0) -> torch.optim.Optimizer:
    """Only trainable parameters are optimized (a frozen encoder is skipped).

    ``backbone_lr_mult < 1`` gives ``model.encoder`` a smaller learning rate
    than the head, the usual way to fine-tune a pretrained encoder without
    destroying what it learned.
    """
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    trainable = [p for p in model.parameters() if p.requires_grad]

    encoder = getattr(model, "encoder", None)
    if encoder is not None and float(backbone_lr_mult) != 1.0:
        encoder_ids = {id(p) for p in encoder.parameters()}
        groups = [
            {"params": [p for p in trainable if id(p) in encoder_ids], "lr": lr * float(backbone_lr_mult)},
            {"params": [p for p in trainable if id(p) not in encoder_ids], "lr": lr},   # the head
        ]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [{"params": trainable, "lr": lr}]

    name = str(train_cfg.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(train_cfg.get("momentum", 0.9)),
                               nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Use adamw, adam or sgd.")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict, epochs: int):
    """Per-epoch schedule (call ``scheduler.step()`` once after each epoch), or None."""
    name = str(train_cfg.get("scheduler") or "none").lower()
    warmup = int(train_cfg.get("warmup_epochs") or 0)

    if name == "cosine":
        main = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1),
                                              eta_min=float(train_cfg.get("min_lr", 0.0)))
    elif name == "none":
        main = None
    else:
        raise ValueError(f"Unknown scheduler '{name}'. Use cosine or none.")

    if warmup <= 0:
        return main
    warm = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    if main is None:
        return warm
    return lr_scheduler.SequentialLR(optimizer, [warm, main], milestones=[warmup])
