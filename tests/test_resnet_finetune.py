"""Partial encoder training and weights-only initialization; no downloads."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.config import Config, load_config
from src.models import build_classifier
from src.tasks import classification
from src.tracking.noop_tracker import NoopTracker
from src.training.optim import build_optimizer
from src.utils.checkpoint import save_checkpoint


@pytest.mark.parametrize("freeze_bn_stats", [False, True])
def test_layer4_updates_but_earlier_weights_and_buffers_stay_fixed(freeze_bn_stats):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        cfg = load_config("configs/cls_resnet50_finetune.yaml")
        cfg.model.params.freeze_bn_stats = freeze_bn_stats
        model = build_classifier(cfg.model)
        model.eval()
        assert not model.encoder.model.layer4.training
        model.train()
        backbone = model.encoder.model
        assert not backbone.bn1.training and not backbone.layer3.training
        assert backbone.layer4.training and model.head.training
        assert backbone.layer4[0].bn1.training is (not freeze_bn_stats)
        for name, parameter in backbone.named_parameters():
            assert parameter.requires_grad == name.startswith("layer4.")

        frozen = {name: value.clone() for name, value in backbone.state_dict().items()
                  if not name.startswith("layer4.")}
        stage_before = backbone.layer4[0].conv1.weight.detach().clone()
        bn_before = {name: value.clone() for name, value in backbone.named_buffers()}
        bn_weight_before = backbone.layer4[0].bn1.weight.detach().clone()
        head_before = model.head[-1].weight.detach().clone()
        optimizer = build_optimizer(model, cfg.train, cfg.model.backbone_lr_mult)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-5, 1e-4])
        logits = model(torch.randn(2, 3, 64, 64))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        assert all(p.grad is None for name, p in backbone.named_parameters()
                   if not name.startswith("layer4."))
        optimizer.step()
        assert not torch.equal(stage_before, backbone.layer4[0].conv1.weight)
        assert not torch.equal(head_before, model.head[-1].weight)
        assert all(torch.equal(value, backbone.state_dict()[name]) for name, value in frozen.items())
        buffers = dict(backbone.named_buffers())
        if freeze_bn_stats:
            assert all(torch.equal(value, buffers[name]) for name, value in bn_before.items())
            assert not torch.equal(bn_weight_before, backbone.layer4[0].bn1.weight)
        else:
            assert not torch.equal(bn_before["layer4.0.bn1.running_mean"],
                                   buffers["layer4.0.bn1.running_mean"])
        model.eval()
        model.train()
        assert backbone.layer4[0].bn1.training is (not freeze_bn_stats)
    finally:
        torch.set_num_threads(previous_threads)


def test_invalid_or_conflicting_stage_selection():
    with pytest.raises(ValueError, match="freeze_backbone"):
        build_classifier({"name": "timm_encoder", "num_classes": 200,
                          "freeze_backbone": True, "params": {"trainable_stages": ["layer4"]}})
    with pytest.raises(ValueError, match="Unknown stage"):
        build_classifier({"name": "timm_encoder", "num_classes": 200,
                          "params": {"backbone": "resnet18", "pretrained": False,
                                     "trainable_stages": ["layer5"]}})


@pytest.mark.parametrize("case", ["initialize", "wrong_labels", "resume"])
def test_checkpoint_initialization_starts_fresh(tmp_path, monkeypatch, case):
    cfg = Config({"device": "cpu", "model": {
        "name": "scratch_cnn", "num_classes": 2, "params": {"depth": 2, "width": 4}},
        "train": {"epochs": 15, "lr": 1e-4, "scheduler": "cosine", "amp": False}})
    source_model = build_classifier(cfg.model)
    with torch.no_grad():
        source_model.head[-1].weight.fill_(0.125)
    source = save_checkpoint(tmp_path / "baseline.pt", source_model, epoch=16,
                             extra={"class_names": ["a", "b"], "best_val_acc": 0.68})
    cfg.train.init_checkpoint = str(source)
    names = ["b", "a"] if case == "wrong_labels" else ["a", "b"]
    loader = SimpleNamespace(dataset=SimpleNamespace(class_names=names))
    monkeypatch.setattr(classification, "build_dataloaders", lambda _: {"train": loader, "val": loader})

    class RecordingTrainer(classification.Trainer):
        def fit(self, resume_from=None):
            if case == "resume":
                assert resume_from == "finetune/last.pt"
                return {"resumed": True}
            assert self.start_epoch == 1 and self.history == [] and self.best_acc == -1
            assert len(self.optimizer.state) == 0
            assert self.optimizer.param_groups[-1]["lr"] == pytest.approx(1e-4)
            for name, value in self.model.state_dict().items():
                assert torch.equal(value, source_model.state_dict()[name])
            return {"initialized": True}

    monkeypatch.setattr(classification, "Trainer", RecordingTrainer)
    if case == "resume":
        cfg.train.resume = "finetune/last.pt"
        cfg.train.init_checkpoint = str(tmp_path / "unavailable-original-baseline.pt")
    task = classification.ClassificationTask(cfg)
    if case == "wrong_labels":
        with pytest.raises(ValueError, match="species-label mapping"):
            task.train(NoopTracker(), tmp_path / "run")
    else:
        result = task.train(NoopTracker(), tmp_path / "run")
        assert result == ({"resumed": True} if case == "resume" else {"initialized": True})


def test_initial_validation_runs_before_weight_updates(tmp_path):
    model = torch.nn.Linear(2, 2)
    initial_weights = model.weight.detach().clone()
    generator = torch.Generator().manual_seed(42)
    generator_state = generator.get_state().clone()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
        torch.tensor([0, 1, 0, 1])), batch_size=4, generator=generator)
    cfg = Config({"train": {"epochs": 1, "lr": 0.01, "amp": False,
                            "validate_before_training": True, "progress": False}})
    tracker = MagicMock()
    calls = []

    class CheckingTrainer(classification.Trainer):
        def _run_epoch(self, loader, train, desc):
            if not calls:
                assert train is False
                assert torch.equal(initial_weights, self.model.weight)
            elif len(calls) == 1:
                assert torch.equal(generator_state, generator.get_state())
                assert torch.equal(random_state, torch.get_rng_state())
            calls.append(train)
            result = super()._run_epoch(loader, train, desc)
            if len(calls) == 1:
                assert torch.equal(initial_weights, self.model.weight)
            return result

    trainer = CheckingTrainer(model, loader, loader, cfg, tracker, tmp_path, torch.device("cpu"))
    random_state = torch.get_rng_state().clone()
    trainer.fit()
    assert calls == [False, True, False]
    assert not torch.equal(initial_weights, model.weight)
    initial_log = tracker.log_metrics.call_args_list[0]
    assert initial_log.kwargs == {"step": 0}
    assert "initial/val_acc" in initial_log.args[0]
