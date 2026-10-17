"""Tracker interface.

Training code talks only to this abstract class, never to W&B or MLflow
directly (Dependency Inversion). Switching tools is then a one-line config
change (``tracker.name: local`` -> ``tracker.name: wandb``), and tests can run
with the ``noop`` tracker without any account or internet connection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Tracker(ABC):
    """What every experiment tracker must be able to do."""

    @abstractmethod
    def start(self, run_name: str, config: dict, run_dir: Path) -> None:
        """Open a run and record its full (resolved) config."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics, e.g. ``{"train/loss": 0.8, "val/acc": 0.41}``."""

    @abstractmethod
    def log_artifact(self, path: str | Path, name: str | None = None, kind: str = "file") -> None:
        """Attach a file (checkpoint, figure, predictions CSV) to the run."""

    def log_summary(self, values: dict) -> None:
        """Final values for the run (best val acc, test metrics...). Optional."""

    def log_figure(self, path: str | Path, name: str | None = None) -> None:
        """Attach a saved image. Defaults to a plain artifact."""
        self.log_artifact(path, name=name, kind="figure")

    @abstractmethod
    def finish(self) -> None:
        """Close the run (flush files, upload pending data)."""

    # Allows ``with build_tracker(cfg) as tracker: ...``
    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
