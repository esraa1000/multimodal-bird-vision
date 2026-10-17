"""A tracker that does nothing; used for tests and quick debugging."""

from __future__ import annotations

from pathlib import Path

from src.registry import TRACKERS
from src.tracking.base import Tracker


@TRACKERS.register("noop")
class NoopTracker(Tracker):
    def start(self, run_name: str, config: dict, run_dir: Path) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str | Path, name: str | None = None, kind: str = "file") -> None:
        pass

    def finish(self) -> None:
        pass
