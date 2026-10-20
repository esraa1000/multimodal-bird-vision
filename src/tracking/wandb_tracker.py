"""Weights & Biases tracking with local records kept beside each run."""

from __future__ import annotations

import importlib
from pathlib import Path
import re

from src.registry import TRACKERS
from src.tracking.local_tracker import LocalTracker


@TRACKERS.register("wandb")
class WandbTracker(LocalTracker):
    """Use the same interface as LocalTracker, with online or offline W&B logging.

    Import the SDK only when starting a run so local/noop tracking does not
    require W&B to be installed. Authentication is handled by ``wandb login``
    or WANDB_API_KEY, never by a YAML config.
    """

    def __init__(
        self,
        project: str = "multimodal-bird-vision",
        entity: str | None = None,
        mode: str = "online",
        group: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        print_every: int = 1,
    ):
        super().__init__(print_every=print_every)
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be online, offline, or disabled")
        self._options = dict(project=project, entity=entity, mode=mode,
                             group=group, tags=tags, notes=notes)
        self._wandb = None
        self._run = None

    def _active_run(self):
        if self._run is None:
            raise RuntimeError("Call tracker.start() before logging to W&B")
        return self._run

    def start(self, run_name: str, config: dict, run_dir: Path) -> None:
        if self._run is not None:
            raise RuntimeError("This W&B tracker already has an active run")
        try:
            self._wandb = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            if exc.name != "wandb":
                raise
            raise ImportError("W&B tracking requires: pip install wandb") from exc
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        # Each start is an independent experiment, including repeated runs in a process.
        self._summary = {}
        self._artifacts = []
        self._calls = 0
        super().start(run_name, config, run_dir)
        self._run = self._wandb.init(
            name=run_name, config=config, dir=str(run_dir), **self._options
        )
        # Keep the caller's epoch/step independent of image logging, which also
        # advances W&B's internal history counter.
        self._run.define_metric("trainer/step")
        self._run.define_metric("*", step_metric="trainer/step", step_sync=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        run = self._active_run()
        super().log_metrics(metrics, step)
        record = dict(metrics)
        if step is not None:
            record["trainer/step"] = step
        run.log(record)

    def log_summary(self, values: dict) -> None:
        run = self._active_run()
        super().log_summary(values)
        run.summary.update(values)

    def log_artifact(self, path: str | Path, name: str | None = None, kind: str = "file") -> None:
        run = self._active_run()
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        # W&B artifact names allow letters, digits, underscores, hyphens and dots.
        artifact_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name or path.name)
        artifact = self._wandb.Artifact(f"{run.id}-{artifact_name}", type=kind)
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        super().log_artifact(path, name, kind)

    def log_figure(self, path: str | Path, name: str | None = None) -> None:
        run = self._active_run()
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Figure file not found: {path}")
        run.log({name or path.stem: self._wandb.Image(str(path))})
        self.log_artifact(path, name, kind="figure")

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        run = self._run
        try:
            super().finish()
            run.summary.update(self._summary)
        finally:
            try:
                run.finish(exit_code=exit_code)
            finally:
                self._run = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish(exit_code=1 if exc_type is not None else 0)
