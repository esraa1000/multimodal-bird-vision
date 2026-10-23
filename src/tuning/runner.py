"""Run this machine's share of a search: one ordinary training run per trial.

Each trial is the tuning config plus that trial's values, named
``<tune.name>_t03``, logged to the W&B group ``<tune.name>``, and recorded as
one line in ``<paths.output_dir>/<tune.name>/results.jsonl``. Running the same
shard again skips trials already recorded as finished, so an interrupted Colab
session just reruns the same command.
"""

from __future__ import annotations

import copy
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Callable

import torch

from src.config import Config
from src.tuning.search import make_trials, select_shard
from src.utils.paths import resolve_path


def results_path(cfg: Config) -> Path:
    return resolve_path(cfg.get_path("paths.output_dir", "outputs")) / cfg.tune.name / "results.jsonl"


def read_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trial_config(cfg: Config, trial: dict) -> Config:
    """The run config for one trial (the ``tune`` section becomes ``tune_trial``)."""
    tune = cfg.tune
    c = Config(copy.deepcopy(cfg.to_dict()))
    c.pop("tune")
    for key, value in trial["params"].items():
        c.set_path(key, value)
    c.run_name = f"{tune.name}_t{trial['trial_id']:02d}"
    c.set_path("paths.output_dir", str(results_path(cfg).parent))
    c.set_path("tracker.group", tune.name)
    c.set_path("tracker.tags", [*(c.get_path("tracker.tags") or []), f"tune-{tune.method}"])
    c.tune_trial = {"name": tune.name, "method": tune.method,
                    "trial_id": trial["trial_id"], "params": dict(trial["params"])}
    return c


def run_tuning(cfg: Config, shard: tuple[int, int], train_fn: Callable[[Config], dict],
               skip_done: bool = True) -> list[dict]:
    """Run the trials of ``shard`` with ``train_fn`` (the CLI's train command)."""
    tune = cfg.tune
    metric = tune.get("metric", "best/val_acc")
    trials = select_shard(make_trials(tune), *shard)
    out_file = results_path(cfg)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = {r["trial_id"] for r in read_results(out_file) if r["status"] == "ok"} if skip_done else set()
    todo = [t for t in trials if t["trial_id"] not in done]
    print(f"[tune] {tune.name} ({tune.method}), shard {shard[0]}/{shard[1]}: {len(trials)} trials, "
          f"{len(trials) - len(todo)} already done. Results -> {out_file}")

    records = []
    for n, trial in enumerate(todo, 1):
        print(f"\n[tune] trial {trial['trial_id']} ({n}/{len(todo)}): {trial['params']}")
        run_cfg = trial_config(cfg, trial)
        record = {"tune": tune.name, "method": tune.method, "trial_id": trial["trial_id"],
                  "params": trial["params"]}
        t0 = time.time()
        try:
            result = train_fn(run_cfg)
            record.update(status="ok", score=result.get(metric), best_epoch=result.get("best/epoch"),
                          time_s=result.get("time/train_s", round(time.time() - t0, 1)))
        except Exception as exc:   # one diverging trial (e.g. lr too high) must not stop the search
            traceback.print_exc()
            record.update(status="failed", score=None, time_s=round(time.time() - t0, 1),
                          error=f"{type(exc).__name__}: {exc}")
        meta = run_cfg.get("meta") or {}
        record["gpu"] = meta.get("gpu", meta.get("device"))
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        records.append(record)
        print(f"[tune] trial {trial['trial_id']}: {record['status']}, {metric}={record['score']}")

        gc.collect()                     # free the finished trial's model before the next one
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records
