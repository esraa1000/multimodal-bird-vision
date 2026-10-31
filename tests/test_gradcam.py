"""Class selectivity, frozen models, state preservation, and real CLI artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.cli import main
from src.config import load_config
from src.models import build_classifier
from src.utils.paths import PROJECT_ROOT
from src.visualization.gradcam import compute_gradcam


class SpatialClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(2, 2, 1, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            self.head.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.head(self.encoder(x).mean(dim=(2, 3)))


def test_gradcam_is_class_specific_and_handles_zero_evidence():
    model = SpatialClassifier().eval().requires_grad_(False)
    x = torch.zeros(1, 2, 8, 8)
    x[:, 0, :, :4], x[:, 1, :, 4:] = 1, 1
    left = compute_gradcam(model, x, 0)["heatmap"]
    right = compute_gradcam(model, x, 1)["heatmap"]
    assert left[:, :4].mean() == 1 and left[:, 4:].mean() == 0
    assert right[:, :4].mean() == 0 and right[:, 4:].mean() == 1
    with torch.no_grad():
        model.head.weight.zero_()
    empty = compute_gradcam(model, x)["heatmap"]
    assert torch.isfinite(empty).all() and empty.count_nonzero() == 0
    with pytest.raises(ValueError, match="class id"):
        compute_gradcam(model, x, 2)
    assert not model.encoder._forward_hooks  # cleaned up on success and failure
    model.train()
    with pytest.raises(ValueError, match="eval"):
        compute_gradcam(model, x)


@pytest.mark.parametrize("config", ["cls_scratch_residual.yaml", "cls_resnet50.yaml", "cls_efficientnet.yaml"])
def test_gradcam_supports_frozen_encoders_without_state_changes(config):
    torch.manual_seed(7)
    cfg = load_config(PROJECT_ROOT / "configs" / config)
    model_cfg = cfg.model.to_dict()
    if model_cfg["name"] == "scratch_cnn":
        model_cfg["params"].update(width=8, depth=2)
    else:
        model_cfg["params"]["pretrained"] = False  # no download in tests
    model = build_classifier(model_cfg).eval().requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(x)
        result = compute_gradcam(model, x)  # locally enables gradients
    torch.testing.assert_close(result["logits"], expected)
    cam = result["heatmap"]
    assert cam.shape == (64, 64) and torch.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert not model.training
    assert all(not module._forward_hooks for module in model.modules())


def test_cli_gradcam_saves_labeled_artifacts_and_preserves_checkpoint(tmp_path, capsys):
    cfg = load_config(PROJECT_ROOT / "configs/cls_smoke.yaml")
    cfg.device = "cpu"
    cfg.model.num_classes = 3
    cfg.data.img_size = 32
    cfg.data.normalize = "imagenet"
    model = build_classifier(cfg.model.to_dict()).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(),
                "extra": {"class_names": ["a", "b", "c"]}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    image = tmp_path / "bird.png"
    pixels = np.full((40, 60, 3), (80, 140, 190), dtype=np.uint8)
    pixels[10:20, 20:30] = 255  # saturated RGB also exercises overlay rounding
    Image.fromarray(pixels).save(image)
    base = ["predict", "--config", str(PROJECT_ROOT / "configs/cls_smoke.yaml"),
            "--ckpt", str(checkpoint), "--image", str(image), "--label", "2", "device=cpu"]
    main(base)
    plain = json.loads(capsys.readouterr().out)
    main(base[:-1] + ["--gradcam", "--gradcam-class", "1", "--output-dir", str(tmp_path / "maps"), "device=cpu"])
    explained = json.loads(capsys.readouterr().out)
    assert plain["prediction"] == explained["prediction"]
    assert plain["confidence"] == explained["confidence"]
    assert explained["correct"] == (explained["prediction_id"] == 2)
    assert explained["gradcam"]["class_id"] == 1
    for key in ("input", "heatmap", "overlay", "comparison"):
        with Image.open(explained["gradcam"][key]) as saved:
            saved.verify()
    cam = np.load(explained["gradcam"]["heatmap_array"])
    assert cam.shape == (32, 32) and np.isfinite(cam).all()
    with Image.open(explained["gradcam"]["input"]) as crop:
        # The inverse normalization recovers the actual model crop's RGB.
        assert np.max(np.abs(np.asarray(crop)[0, 0, :3].astype(int) - [80, 140, 190])) <= 1
    metadata = json.loads(Path(explained["gradcam"]["metadata"]).read_text())
    assert metadata["preprocessing"]["normalize"] == "imagenet"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
