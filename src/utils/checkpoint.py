"""Saving and resuming training state.

A checkpoint holds everything needed to resume a run that was cut off
(e.g. a Colab session that disconnected) or to reload the model for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int = 0,
    metrics: dict | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """``extra`` holds anything else worth keeping (class names, best score so far,
    history, AMP scaler state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "extra": extra or {},
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load weights (and optionally optimizer/scheduler state) in place.

    Returns the full checkpoint dict so callers can read ``epoch``,
    ``metrics`` and ``config``.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    return state
