"""Device selection."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return ``cuda`` / ``mps`` / ``cpu``.

    ``preference="auto"`` picks the best available device; any other value
    (e.g. ``"cpu"``, ``"cuda:1"``) is used as given.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    """Hardware info to log with every run."""
    info = {"device": str(device), "torch": str(torch.__version__)}
    if device.type == "cuda":
        idx = device.index or 0
        info["gpu"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
    return info
