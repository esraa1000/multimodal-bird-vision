"""Grid / random search: trial plans, sharding, the runner and the comparison report."""

import json
from types import SimpleNamespace

import pytest

from src.cli import main
from src.config import Config
from src.tuning import make_trials, parse_shard, run_tuning, select_shard
from src.tuning import report
from src.tuning.runner import read_results, results_path, trial_config
from src.utils.paths import PROJECT_ROOT

GRID = {"name": "g", "method": "grid",
        "space": {"train.lr": [1e-3, 3e-3, 1e-2], "model.dropout": [0.1, 0.5]}}
RANDOM = {"name": "r", "method": "random", "n_trials": 8, "seed": 1,
          "space": {"train.lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
                    "model.dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
                    "model.params.depth": {"type": "int", "low": 2, "high": 4},
                    "data.augment": ["light", "strong"]}}


# ------------------------------------------------------------------ plans

def test_grid_is_every_combination():
    trials = make_trials(GRID)
    assert len(trials) == 6 and [t["trial_id"] for t in trials] == list(range(6))
    assert trials[0]["params"] == {"train.lr": 1e-3, "model.dropout": 0.1}
    assert trials[-1]["params"] == {"train.lr": 1e-2, "model.dropout": 0.5}


def test_random_is_reproducible_and_in_range():
    trials = make_trials(RANDOM)
    assert trials == make_trials(RANDOM)                       # same plan on every machine
    assert trials != make_trials({**RANDOM, "seed": 2})
    for t in trials:
        p = t["params"]
        assert 1e-4 <= p["train.lr"] <= 1e-2 and 0.0 <= p["model.dropout"] <= 0.5
        assert p["model.params.depth"] in (2, 3, 4) and p["data.augment"] in ("light", "strong")


def test_shards_cover_every_trial_once():
    trials = make_trials(GRID)
    ids = [t["trial_id"] for k in range(1, 5) for t in select_shard(trials, k, 4)]
    assert sorted(ids) == list(range(6))
    assert parse_shard("2/4") == (2, 4) and parse_shard(None) == (1, 1)
    with pytest.raises(ValueError):
        parse_shard("5/4")


# ----------------------------------------------------------------- runner

def _cfg(tmp_path, tune):
    return Config({"paths": {"output_dir": str(tmp_path)}, "train": {"lr": 0.0},
                   "model": {"dropout": 0.0}, "tracker": {"name": "noop", "tags": ["x"]}, "tune": tune})


def test_trial_config(tmp_path):
    cfg = _cfg(tmp_path, GRID)
    c = trial_config(cfg, make_trials(GRID)[4])
    assert c.train.lr == 1e-2 and c.model.dropout == 0.1
    assert c.run_name == "g_t04" and c.tracker.group == "g" and c.tracker.tags == ["x", "tune-grid"]
    assert c.tune_trial["trial_id"] == 4 and "tune" not in c
    assert c.paths.output_dir == str(tmp_path / "g")


def test_runner_records_failures_and_skips_finished(tmp_path):
    cfg = _cfg(tmp_path, GRID)
    calls = []

    def fake_train(c):
        calls.append(c.run_name)
        if c.train.lr >= 1e-2:
            raise FloatingPointError("loss became nan")
        return {"best/val_acc": 0.1 + c.train.lr * 10 - c.model.dropout / 10, "time/train_s": 60.0}

    records = run_tuning(cfg, (1, 1), fake_train)
    assert [r["status"] for r in records] == ["ok"] * 4 + ["failed"] * 2
    assert "nan" in records[-1]["error"]
    assert len(read_results(results_path(cfg))) == 6

    calls.clear()
    run_tuning(cfg, (1, 1), fake_train)            # rerun: only the failed trials again
    assert calls == ["g_t04", "g_t05"]


# ----------------------------------------------------------------- report

def _write_results(cfg, rows):
    path = results_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(tune, i, score, status="ok"):
    return {"tune": tune["name"], "method": tune["method"], "trial_id": i, "params": {"train.lr": 10.0 ** -(i + 2)},
            "status": status, "score": score, "time_s": 120.0, "gpu": "Tesla T4"}


def test_report_compares_methods(tmp_path):
    grid, rand = _cfg(tmp_path, GRID), _cfg(tmp_path, {**RANDOM, "n_trials": 3})
    _write_results(grid, [_row(GRID, 0, 0.30), _row(GRID, 1, 0.40), _row(GRID, 2, 0.41), _row(GRID, 3, None, "failed")])
    _write_results(rand, [_row(RANDOM, 0, 0.42), _row(RANDOM, 1, 0.45), _row(RANDOM, 2, 0.20)])

    result = report.build_report([grid, rand], source="local", out_dir=tmp_path / "report")
    g, r = result["methods"]
    assert result["target"] == pytest.approx(0.43)               # best 0.45 minus 2 points
    assert g["best_score"] == 0.41 and g["trials_to_best"] == 3 and g["trials_to_target"] is None
    assert g["trials_failed"] == 1 and g["trials_planned"] == 6 and g["total_time_min"] == 8.0
    assert r["best_score"] == 0.45 and r["trials_to_best"] == 2 and r["trials_to_target"] == 2
    for name in ("tuning_trials.csv", "tuning_comparison.json", "tuning_comparison.png"):
        assert (tmp_path / "report" / name).is_file()


def test_report_from_wandb(tmp_path, monkeypatch):
    def run(i, state, score, created):
        return SimpleNamespace(
            state=state, created_at=created, summary={"best/val_acc": score, "time/train_s": 60.0},
            config={"tune_trial": {"name": "g", "method": "grid", "trial_id": i, "params": {"train.lr": 0.001}},
                    "meta": {"gpu": "Tesla T4"}})

    runs = [run(0, "crashed", None, "2026-10-24T10:00"), run(0, "finished", 0.3, "2026-10-24T11:00"),
            run(1, "finished", 0.5, "2026-10-24T10:00"), run(2, "running", None, "2026-10-24T12:00")]
    api = SimpleNamespace(default_entity="team", runs=lambda path, filters: runs)
    monkeypatch.setattr(report, "_wandb_api", lambda: api)

    cfg = _cfg(tmp_path, GRID)
    cfg.tracker = {"name": "wandb", "project": "p"}
    records = report.dedupe(report.collect_wandb(cfg))
    assert [(r["trial_id"], r["status"], r["score"]) for r in records] == [(0, "ok", 0.3), (1, "ok", 0.5)]


# ------------------------------------------------------------ CLI, end to end

def test_cli_tune_and_report(fake_cub, tmp_path, capsys):
    configs = []
    for name, tune in [("grid", {"name": "tg", "method": "grid", "space": {"train.lr": [1e-3, 3e-3]}}),
                       ("random", {"name": "tr", "method": "random", "n_trials": 2, "seed": 0,
                                   "space": {"train.lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}}})]:
        path = tmp_path / f"tune_{name}.yaml"
        path.write_text(json.dumps({"defaults": (PROJECT_ROOT / "configs" / "tune_base.yaml").as_posix(),
                                    "tune": tune}))         # JSON is valid YAML
        configs.append(str(path))
        main(["tune", "--config", str(path), "--shard", "1/1", *fake_cub.overrides, "train.epochs=1"])

    capsys.readouterr()
    out = tmp_path / "report"
    main(["tune-report", "--config", configs[0], "--config", configs[1], "--source", "local",
          "--out", str(out), *fake_cub.overrides])
    summary = json.loads((out / "tuning_comparison.json").read_text())
    assert [m["trials_completed"] for m in summary["methods"]] == [2, 2]
    assert "best grid" in capsys.readouterr().out
    assert len(list((fake_cub.root / "outputs" / "tg").glob("tg_t0*"))) == 2   # one run folder per trial
