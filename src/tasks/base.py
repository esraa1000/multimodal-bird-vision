"""Task interface: one class per project task (classification, colorization, caption).

The CLI only knows this interface. ``python -m src.cli train --task X`` looks
up task ``X`` in the ``TASKS`` registry and calls its ``train`` method, so
Milestone 2 tasks plug in without changing the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.config import Config
from src.tracking import Tracker


class Task(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def train(self, tracker: Tracker, run_dir: Path) -> dict:
        """Train a model; save checkpoints under ``run_dir``; return final metrics."""

    @abstractmethod
    def evaluate(self, checkpoint: Path, split: str, tracker: Tracker, run_dir: Path) -> dict:
        """Evaluate a trained checkpoint on ``split`` (``val`` or ``test``)."""

    @abstractmethod
    def predict(self, checkpoint: Path, image_path: Path, **options) -> dict:
        """Run inference on a single image and return a JSON-serialisable result."""
