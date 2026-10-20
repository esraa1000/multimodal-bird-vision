"""Tracker contract checks and a real SDK smoke test without network access."""

import json
from unittest.mock import MagicMock

import pytest

from src.config import load_config
from src.tracking import build_tracker


@pytest.fixture
def sdk(monkeypatch):
    sdk = MagicMock()
    sdk.init.return_value.id = "test-run"
    monkeypatch.setattr("src.tracking.wandb_tracker.importlib.import_module", lambda _: sdk)
    return sdk


def test_logs_config_metrics_summary_and_artifacts(tmp_path, sdk):
    tracker = build_tracker({"name": "wandb", "mode": "offline"})
    config = {"seed": 42, "train": {"lr": 0.001}}
    tracker.start("baseline", config, tmp_path)
    artifact_path = tmp_path / "best.pt"
    artifact_path.write_bytes(b"checkpoint")
    figure_path = tmp_path / "curve.png"
    figure_path.write_bytes(b"fake image for mocked SDK")
    with tracker:
        tracker.log_metrics({"val/acc": 0.75}, step=1)
        tracker.log_summary({"best/val_acc": 0.75})
        tracker.log_artifact(artifact_path, name="best model", kind="model")
        tracker.log_figure(figure_path, name="loss curve")
    sdk.init.assert_called_once_with(
        name="baseline", config=config, dir=str(tmp_path.resolve()),
        project="multimodal-bird-vision", entity=None, mode="offline",
        group=None, tags=None, notes=None,
    )
    run = sdk.init.return_value
    assert run.log.call_args_list[0].args == ({"val/acc": 0.75, "trainer/step": 1},)
    run.define_metric.assert_any_call("*", step_metric="trainer/step", step_sync=True)
    sdk.Image.assert_called_once_with(str(figure_path.resolve()))
    assert run.log_artifact.call_count == 2
    sdk.Artifact.assert_any_call("test-run-best_model", type="model")
    run.finish.assert_called_once_with(exit_code=0)
    assert (tmp_path / "config.yaml").is_file()
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["val/acc"] == 0.75
    assert json.loads((tmp_path / "summary.json").read_text())["best/val_acc"] == 0.75
    assert len(json.loads((tmp_path / "artifacts.json").read_text())) == 2
    tracker.finish()  # Finishing twice must not close a different run.
    run.finish.assert_called_once()


def test_exception_marks_run_failed(tmp_path, sdk):
    tracker = build_tracker({"name": "wandb"})
    tracker.start("failure", {}, tmp_path)
    with pytest.raises(ValueError, match="training failed"):
        with tracker:
            raise ValueError("training failed")
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_lifecycle_and_missing_files(tmp_path, sdk):
    tracker = build_tracker({"name": "wandb"})
    with pytest.raises(RuntimeError, match="start"):
        tracker.log_metrics({"loss": 1.0})
    tracker.start("test", {}, tmp_path)
    with tracker:
        with pytest.raises(RuntimeError, match="active run"):
            tracker.start("second", {}, tmp_path)
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(tmp_path / "missing.pt")


def test_missing_sdk_does_not_break_local_tracking(tmp_path, monkeypatch):
    def missing_sdk(_):
        raise ModuleNotFoundError("No module named 'wandb'", name="wandb")

    monkeypatch.setattr("src.tracking.wandb_tracker.importlib.import_module", missing_sdk)
    local = build_tracker({"name": "local"})
    local.start("local", {}, tmp_path)
    with local:
        local.log_metrics({"loss": 1.0})
    with pytest.raises(ImportError, match="pip install wandb"):
        build_tracker({"name": "wandb"}).start("test", {}, tmp_path)


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        build_tracker({"name": "wandb", "mode": "typo"})


def test_example_config():
    cfg = load_config("configs/debug_wandb.yaml")
    assert cfg.task == "debug"
    assert cfg.tracker.name == "wandb"
    assert cfg.tracker.mode == "offline"


def test_real_offline_sdk(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "data"))
    tracker = build_tracker({"name": "wandb", "mode": "offline"})
    tracker.start("offline-smoke", {"seed": 42}, tmp_path)
    with tracker:
        tracker.log_metrics({"train/loss": 0.5, "val/acc": 0.75}, step=1)
        tracker.log_summary({"best/val_acc": 0.75})
        checkpoint = tmp_path / "checkpoint.txt"
        checkpoint.write_text("test checkpoint")
        tracker.log_artifact(checkpoint, kind="model")
    assert list((tmp_path / "wandb").glob("offline-run-*/*.wandb"))
    assert json.loads((tmp_path / "summary.json").read_text())["best/val_acc"] == 0.75
