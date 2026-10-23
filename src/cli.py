"""Command line entry point.

Examples::

    python -m src.cli prepare-data
    python -m src.cli train    --task classification --config configs/cls_scratch.yaml
    python -m src.cli train    --config configs/cls_scratch.yaml train.lr=1e-3 train.epochs=5
    python -m src.cli train    --config configs/cls_scratch.yaml --resume outputs/<run>/last.pt
    python -m src.cli evaluate --config configs/cls_scratch.yaml --ckpt outputs/<run>/best.pt --split test
    python -m src.cli predict  --config configs/cls_scratch.yaml --ckpt outputs/<run>/best.pt --image bird.jpg
    python -m src.cli tune     --config configs/tune_grid.yaml --shard 1/4
    python -m src.cli tune-report --config configs/tune_grid.yaml --config configs/tune_random.yaml
    python -m src.cli list

Anything after the named arguments in the form ``key.subkey=value`` overrides
the config file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import Config, load_config
from src.registry import MODELS, TASKS, TRACKERS
from src.tracking import build_tracker
from src.utils.device import describe_device, get_device
from src.utils.paths import detect_environment, get_data_dir, git_commit_hash, make_run_dir
from src.utils.seed import seed_everything

import src.tasks  # noqa: F401,E402  (registers tasks)


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, args.overrides)
    if getattr(args, "task", None):
        cfg.task = args.task
    if "task" not in cfg:
        raise SystemExit("No task given: pass --task or set 'task:' in the config.")
    return cfg


def _build_task(cfg: Config):
    return TASKS.build(cfg.task, cfg)


def _start_run(cfg: Config, mode: str):
    """Seed, create the run folder, open the tracker with full run metadata."""
    seed_everything(cfg.get("seed", 42), cfg.get("deterministic", False))
    if mode == "train":
        run_name = cfg.get("run_name") or f"{cfg.task}_train"
    else:   # e.g. "cls_scratch_eval-test", so evaluations are not mistaken for training runs
        run_name = f"{cfg.get('run_name') or cfg.task}_{mode}"
    run_dir = make_run_dir(cfg.get_path("paths.output_dir", "outputs"), run_name)

    cfg.meta = {
        "mode": mode,
        "git_commit": git_commit_hash(),
        "environment": detect_environment(),
        **describe_device(get_device(cfg.get("device", "auto"))),
    }
    tracker = build_tracker(cfg.get("tracker", {"name": "local"}))
    tracker.start(run_name, cfg.to_dict(), run_dir)
    return tracker, run_dir


def _train(cfg: Config) -> dict:
    """One tracked training run (used by ``train`` and by every ``tune`` trial)."""
    tracker, run_dir = _start_run(cfg, "train")
    with tracker:
        return _build_task(cfg).train(tracker, run_dir)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load(args)
    if args.resume:
        cfg.set_path("train.resume", args.resume)
    print(json.dumps(_train(cfg), indent=2))


def cmd_tune(args: argparse.Namespace) -> None:
    from src.tuning import make_trials, parse_shard, run_tuning, select_shard

    cfg = _load(args)
    if "tune" not in cfg:
        raise SystemExit(f"{args.config} has no 'tune' section (see configs/tune_grid.yaml).")
    shard = parse_shard(args.shard)
    if args.list:
        trials = select_shard(make_trials(cfg.tune), *shard)
        print(f"{cfg.tune.name} ({cfg.tune.method}), shard {shard[0]}/{shard[1]}: {len(trials)} trials")
        for t in trials:
            print(f"  trial {t['trial_id']:>3}: {t['params']}")
        return
    records = run_tuning(cfg, shard, _train, skip_done=not args.rerun)
    print(json.dumps(records, indent=2))


def cmd_tune_report(args: argparse.Namespace) -> None:
    from src.tuning.report import build_report

    cfgs = [load_config(path, args.overrides) for path in args.config]
    build_report(cfgs, source=args.source, out_dir=args.out)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _load(args)
    tracker, run_dir = _start_run(cfg, f"eval-{args.split}")
    with tracker:
        result = _build_task(cfg).evaluate(Path(args.ckpt), args.split, tracker, run_dir)
    print(json.dumps(result, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = _load(args)
    seed_everything(cfg.get("seed", 42))
    options = {"gradcam": args.gradcam}
    result = _build_task(cfg).predict(Path(args.ckpt), Path(args.image), **options)
    print(json.dumps(result, indent=2))


def cmd_prepare_data(args: argparse.Namespace) -> None:
    from src.data.download import prepare_cub

    cfg = load_config(args.config, args.overrides)
    data_dir = get_data_dir(cfg.get("paths", {}))
    prepare_cub(data_dir, force=args.force)


def cmd_list(args: argparse.Namespace) -> None:
    import src.models  # noqa: F401  (models register on import)

    print(f"tasks:    {TASKS.keys()}")
    print(f"models:   {MODELS.keys()}")
    print(f"trackers: {TRACKERS.keys()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, needs_task: bool = True) -> None:
        p.add_argument("--config", default="configs/base.yaml", help="YAML config file")
        if needs_task:
            p.add_argument("--task", choices=TASKS.keys(), help="overrides 'task' in the config")
        p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.lr=1e-3")

    p = sub.add_parser("train", help="train a model")
    add_common(p)
    p.add_argument("--resume", help="last.pt of an interrupted run, to continue training from it")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    add_common(p)
    p.add_argument("--ckpt", required=True, help="path to a .pt checkpoint")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="run inference on one image")
    add_common(p)
    p.add_argument("--ckpt", required=True, help="path to a .pt checkpoint")
    p.add_argument("--image", required=True, help="path to an image file")
    p.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM heatmap")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("tune", help="run (a shard of) a grid or random hyperparameter search")
    add_common(p)
    p.add_argument("--shard", help="k/n: run every n-th trial starting at the k-th, e.g. 2/4")
    p.add_argument("--list", action="store_true", help="only print the planned trials")
    p.add_argument("--rerun", action="store_true", help="also rerun trials already finished")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("tune-report", help="compare finished searches (e.g. grid vs. random)")
    p.add_argument("--config", action="append", required=True,
                   help="a tuning config; repeat for each search to compare")
    p.add_argument("--source", default="wandb", choices=["wandb", "local"],
                   help="read trials from W&B (whole team) or local results.jsonl files")
    p.add_argument("--out", default="reports/tuning", help="folder for the CSV, JSON and figure")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. paths.output_dir=...")
    p.set_defaults(func=cmd_tune_report)

    p = sub.add_parser("prepare-data", help="download and cache the dataset")
    add_common(p, needs_task=False)
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("list", help="show registered tasks, models and trackers")
    p.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
