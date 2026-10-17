"""Placeholders for the real tasks so the CLI already lists them.

Each owner replaces their placeholder with a real implementation in its own
module (e.g. ``src/tasks/classification.py``) and deletes it from here.
"""

from __future__ import annotations

from pathlib import Path

from src.registry import TASKS
from src.tasks.base import Task


class _NotImplementedTask(Task):
    owner = "?"
    milestone = "?"

    def _fail(self):
        raise NotImplementedError(
            f"Task '{self.cfg.get('task')}' is not implemented yet "
            f"(owner: {self.owner}, milestone {self.milestone})."
        )

    def train(self, tracker, run_dir: Path) -> dict:
        self._fail()

    def evaluate(self, checkpoint, split, tracker, run_dir) -> dict:
        self._fail()

    def predict(self, checkpoint, image_path, **options) -> dict:
        self._fail()


@TASKS.register("classification")
class ClassificationTask(_NotImplementedTask):
    owner, milestone = "M3/M4", "1"


@TASKS.register("colorization")
class ColorizationTask(_NotImplementedTask):
    owner, milestone = "TBD", "2"


@TASKS.register("caption")
class CaptionTask(_NotImplementedTask):
    owner, milestone = "TBD", "2"
