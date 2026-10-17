"""Offline tracker that writes everything into the run directory.

Works with no account or internet. Each run folder gets::

    outputs/<run>_<time>/
    ├── config.yaml      # resolved config + git hash + hardware
    ├── metrics.jsonl    # one JSON line per log_metrics() call
    ├── summary.json     # final/best values
    └── artifacts.json   # list of files attached to the run

The W&B tracker (added later) implements the same interface.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import Config, save_config
from src.registry import TRACKERS
from src.tracking.base import Tracker


@TRACKERS.register("local")
class LocalTracker(Tracker):
    def __init__(self, print_every: int = 1, **_: object):
        self.print_every = print_every
        self.run_dir: Path | None = None
        self._summary: dict = {}
        self._artifacts: list[dict] = []
        self._calls = 0
        self._t0 = 0.0

    def start(self, run_name: str, config: dict, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._t0 = time.time()
        save_config(Config(config), self.run_dir / "config.yaml")
        print(f"[local tracker] run '{run_name}' -> {self.run_dir}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        record = {"step": step, "time_s": round(time.time() - self._t0, 2), **metrics}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._calls += 1
        if self._calls % self.print_every == 0:
            shown = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            print(f"[step {step}] {shown}")

    def log_artifact(self, path: str | Path, name: str | None = None, kind: str = "file") -> None:
        self._artifacts.append({"name": name or Path(path).name, "kind": kind, "path": str(path)})

    def log_summary(self, values: dict) -> None:
        self._summary.update(values)

    def finish(self) -> None:
        if self.run_dir is None:
            return
        self._summary["runtime_s"] = round(time.time() - self._t0, 2)
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(self._summary, f, indent=2)
        with open(self.run_dir / "artifacts.json", "w", encoding="utf-8") as f:
            json.dump(self._artifacts, f, indent=2)
        self.run_dir = None
