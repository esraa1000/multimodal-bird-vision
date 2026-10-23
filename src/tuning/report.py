"""Compare tuning methods (e.g. grid vs. random search) on the same budget.

    python -m src.cli tune-report --config configs/tune_grid.yaml --config configs/tune_random.yaml

Trials are read from W&B (``--source wandb``: everyone's trials, found by the
W&B group ``tune.name``) or from the local ``results.jsonl`` files written by
``tune`` (``--source local``). Writes to ``--out``:

* ``tuning_trials.csv``: one row per trial (method, values, score, time, GPU)
* ``tuning_comparison.json``: per method, the numbers below
* ``tuning_comparison.png``: best score so far vs. trial number, and score vs. one hyperparameter

Per method: best score and its values, trials completed / failed, trials
needed to reach the best and to come within ``target_gap`` (default 2
accuracy points, about the noise of a 599-image validation set) of the overall
best, and total / mean training time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.config import Config
from src.tuning.runner import read_results, results_path
from src.tuning.search import make_trials
from src.utils.paths import resolve_path
from src.visualization.plots import plot_tuning_comparison


def _wandb_api():
    import wandb

    return wandb.Api()


def collect_local(cfg: Config) -> list[dict]:
    return read_results(results_path(cfg))


def collect_wandb(cfg: Config) -> list[dict]:
    api = _wandb_api()
    tracker = cfg.get("tracker", {})
    entity = tracker.get("entity") or api.default_entity
    path = f"{entity}/{tracker.get('project', 'multimodal-bird-vision')}"
    records = []
    for run in api.runs(path, filters={"group": cfg.tune.name}):
        trial = run.config.get("tune_trial")
        if not trial or run.state == "running":
            continue
        ok = run.state == "finished"
        records.append({
            "tune": trial["name"], "method": trial["method"], "trial_id": trial["trial_id"],
            "params": trial["params"], "status": "ok" if ok else "failed",
            "score": run.summary.get(cfg.tune.get("metric", "best/val_acc")) if ok else None,
            "best_epoch": run.summary.get("best/epoch"), "time_s": run.summary.get("time/train_s"),
            "gpu": (run.config.get("meta") or {}).get("gpu"), "created_at": str(run.created_at),
        })
    return records


def dedupe(records: list[dict]) -> list[dict]:
    """One record per trial: a finished one beats a failed one, a later one beats an earlier one."""
    by_trial: dict[int, dict] = {}
    for r in sorted(records, key=lambda r: r.get("created_at", "")):
        prev = by_trial.get(r["trial_id"])
        if prev is None or r["status"] == "ok" or prev["status"] != "ok":
            by_trial[r["trial_id"]] = r
    return sorted(by_trial.values(), key=lambda r: r["trial_id"])


def finished(records: list[dict]) -> list[dict]:
    return [r for r in records if r["status"] == "ok" and r.get("score") is not None]


def summarize(tune: dict, records: list[dict], target: float) -> dict:
    ok = finished(records)
    times = [r.get("time_s") or 0.0 for r in records]
    summary = {
        "name": tune["name"], "method": tune["method"],
        "trials_planned": len(make_trials(tune)), "trials_completed": len(ok),
        "trials_failed": len(records) - len(ok),
        "total_time_min": round(sum(times) / 60, 1),
        "mean_trial_time_min": round(float(np.mean(times)) / 60, 1) if times else None,
        "gpus": sorted({r["gpu"] for r in records if r.get("gpu")}),
    }
    if ok:
        scores = np.array([r["score"] for r in ok])
        best_i = int(scores.argmax())
        reached = next((i + 1 for i, s in enumerate(scores) if s >= target), None)
        summary.update(
            best_score=float(scores[best_i]), best_trial_id=ok[best_i]["trial_id"],
            best_params=ok[best_i]["params"], mean_score=float(scores.mean()), std_score=float(scores.std()),
            trials_to_best=best_i + 1, trials_to_target=reached,
        )
    return summary


def build_report(cfgs: list[Config], source: str = "wandb", out_dir: str | Path = "reports/tuning",
                 target_gap: float = 0.02) -> dict:
    per_method = []
    for cfg in cfgs:
        records = collect_wandb(cfg) if source == "wandb" else collect_local(cfg)
        per_method.append((cfg, dedupe(records)))

    scores = [r["score"] for _, recs in per_method for r in finished(recs)]
    if not scores:
        raise SystemExit(f"No finished trials found (source: {source}).")
    target = max(scores) - target_gap
    summaries = [summarize(cfg.tune, recs, target) for cfg, recs in per_method]

    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_trials_csv(out / "tuning_trials.csv", [r for _, recs in per_method for r in recs])
    with open(out / "tuning_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"source": source, "target": target,
                   "target_rule": f"overall best score - {target_gap}",
                   "methods": summaries}, f, indent=2)
    first_param = next(iter(cfgs[0].tune.space))
    plot_tuning_comparison({f"{c.tune.method} ({c.tune.name})": finished(recs) for c, recs in per_method},
                           out / "tuning_comparison.png", param=first_param)
    _print_table(summaries, target)
    return {"target": target, "methods": summaries, "out_dir": str(out)}


def _write_trials_csv(path: Path, records: list[dict]) -> None:
    params = sorted({k for r in records for k in r["params"]})
    columns = ["tune", "method", "trial_id", "status", "score", "best_epoch", "time_s", "gpu", *params]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in records:
            writer.writerow([r.get(c) for c in columns[:8]] + [r["params"].get(p) for p in params])


def _print_table(summaries: list[dict], target: float) -> None:
    print(f"\n{'method':<8} {'name':<18} {'done':>7} {'failed':>6} {'best':>7} {'to best':>7} "
          f"{'to target':>9} {'time(min)':>9}   (target = {target:.4f})")
    for s in summaries:
        done = f"{s['trials_completed']}/{s['trials_planned']}"
        best = f"{s['best_score']:.4f}" if "best_score" in s else "-"
        print(f"{s['method']:<8} {s['name']:<18} {done:>7} {s['trials_failed']:>6} {best:>7} "
              f"{s.get('trials_to_best', '-')!s:>7} {s.get('trials_to_target') or '-'!s:>9} "
              f"{s['total_time_min']:>9}")
    for s in summaries:
        if "best_params" in s:
            print(f"  best {s['method']}: {s['best_params']}")
