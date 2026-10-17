"""Experiment tracking behind a single interface (see ``base.Tracker``)."""

from src.registry import TRACKERS
from src.tracking import local_tracker, noop_tracker  # noqa: F401  (registers trackers)
from src.tracking.base import Tracker


def build_tracker(tracker_cfg: dict) -> Tracker:
    """Build the tracker named in ``cfg.tracker`` (``noop``, ``local``, later ``wandb``)."""
    cfg = dict(tracker_cfg or {})
    name = cfg.pop("name", "local")
    return TRACKERS.build(name, **cfg)


__all__ = ["Tracker", "build_tracker"]
