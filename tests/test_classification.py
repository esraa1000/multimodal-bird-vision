"""Classification pipeline checks on CPU, without downloading CUB.

The end-to-end test builds a tiny fake CUB in the same Hugging Face format
(image / label / file_name), so it runs the real data loaders, model, trainer,
CLI, evaluation and prediction code.
"""

import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.cli import main
from src.config import Config
from src.evaluation.metrics import classification_metrics, confusion_matrix, top_confused_pairs
from src.models import Classifier, build_classifier
from src.models.scratch_cnn import ScratchCNN
from src.tracking.noop_tracker import NoopTracker
from src.training import Trainer
from src.utils.paths import PROJECT_ROOT

CPU = torch.device("cpu")


# ----------------------------------------------------------------------- model

@pytest.mark.parametrize("residual", [False, True])
def test_scratch_cnn_shapes(residual):
    enc = ScratchCNN(depth=3, width=8, residual=residual)
    x = torch.randn(2, 3, 64, 64)
    assert enc.out_dim == 32
    assert enc.forward_features(x).shape == (2, 32, 8, 8)     # stem /2, then two pools /4
    assert enc(x).shape == (2, 32)
    assert [m.shape[1] for m in enc.forward_stages(x)] == [8, 16, 32]


def test_scratch_cnn_grayscale_input():
    assert ScratchCNN(depth=2, width=8, in_channels=1)(torch.randn(1, 1, 32, 32)).shape == (1, 16)


def test_build_classifier_and_freeze():
    model = build_classifier({"name": "scratch_cnn", "num_classes": 7, "dropout": 0.2,
                              "freeze_backbone": True, "params": {"depth": 2, "width": 8}})
    assert model(torch.randn(3, 3, 32, 32)).shape == (3, 7)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.train()
    assert model.head.training and not model.encoder.training   # BatchNorm stays fixed


# --------------------------------------------------------------------- metrics

def test_classification_metrics():
    y = np.array([0, 1, 2, 2])
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3], [0.1, 0.1, 0.8]])
    m = classification_metrics(y, probs)
    assert m["acc"] == pytest.approx(0.75)
    assert m["top5_acc"] == 1.0        # only 3 classes, so the true one is always in the top k
    assert m["num_images"] == 4
    cm = confusion_matrix(y, probs.argmax(1), 3)
    assert cm[2, 0] == 1 and cm.trace() == 3
    assert top_confused_pairs(cm, ["a", "b", "c"]) == [
        {"true_id": 2, "true_name": "c", "pred_id": 0, "pred_name": "a", "count": 1}]


# --------------------------------------------------------------------- trainer

def _separable_loader(n=32, classes=4):
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n) % classes
    x = torch.randn(n, 3, 16, 16, generator=g) + y.view(-1, 1, 1, 1).float()   # class = brightness
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g)


def _trainer(tmp_path, epochs, lr=1e-2, **train):
    torch.manual_seed(0)
    cfg = Config({"train": {"epochs": epochs, "lr": lr, "optimizer": "adamw", "scheduler": "cosine",
                            "amp": False, "progress": False, **train}})
    model = Classifier(ScratchCNN(depth=2, width=8), num_classes=4)
    loader = _separable_loader()
    return Trainer(model, loader, loader, cfg, NoopTracker(), tmp_path, CPU)


def test_trainer_learns_and_saves(tmp_path):
    result = _trainer(tmp_path, epochs=8, warmup_epochs=1).fit()
    assert result["best/val_acc"] > 0.6      # chance is 0.25
    for name in ("best.pt", "last.pt", "training_curves.png"):
        assert (tmp_path / name).is_file()
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert best["optimizer_state"] is None and last["optimizer_state"] is not None
    assert len(last["extra"]["history"]) == 8


def test_trainer_resume_continues(tmp_path):
    first = tmp_path / "first"
    _trainer(first, epochs=2).fit()
    resumed = _trainer(tmp_path / "resumed", epochs=4)
    resumed.fit(resume_from=first / "last.pt")
    assert [h["epoch"] for h in resumed.history] == [1, 2, 3, 4]
    assert (tmp_path / "resumed" / "best.pt").is_file()


def test_early_stopping(tmp_path):
    trainer = _trainer(tmp_path, epochs=10, lr=0.0, early_stopping_patience=2)   # lr 0: never improves
    result = trainer.fit()
    assert result["stopped_early"] and result["epochs_trained"] == 3


# ------------------------------------------------------------ CLI, end to end

CLASSES = ["Albatross", "Cardinal", "Jay", "Wren"]


@pytest.fixture
def fake_cub(tmp_path):
    from datasets import ClassLabel, Dataset, Features, Value
    from datasets import Image as HFImage

    rng = np.random.default_rng(0)
    features = Features({"image": HFImage(), "label": ClassLabel(names=CLASSES),
                         "file_name": Value("string")})

    def make(split, per_class):
        rows = {"image": [], "label": [], "file_name": []}
        for c, name in enumerate(CLASSES):
            for i in range(per_class):
                colour = np.array([c * 60, 255 - c * 60, 128])          # each class has its own colour
                pixels = colour + rng.integers(-20, 20, size=(40, 48, 3))
                rows["image"].append(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)))
                rows["label"].append(c)
                rows["file_name"].append(f"{name}_{split}_{i:03d}.jpg")
        Dataset.from_dict(rows, features=features).save_to_disk(str(tmp_path / "data" / "cub" / split))
        return rows["file_name"]

    train_names = make("train", 6)
    make("test", 3)
    is_val = lambda n: n.endswith(("_004.jpg", "_005.jpg"))  # noqa: E731
    (tmp_path / "split.json").write_text(json.dumps({
        "train": [n for n in train_names if not is_val(n)], "val": [n for n in train_names if is_val(n)]}))
    Image.fromarray(np.full((40, 48, 3), (0, 255, 128), dtype=np.uint8)).save(tmp_path / "bird.jpg")
    return tmp_path


def _overrides(root):
    return [
        f"paths.data_dir={(root / 'data').as_posix()}",
        f"paths.output_dir={(root / 'outputs').as_posix()}",
        f"data.split_file={(root / 'split.json').as_posix()}",
        "tracker.name=local", "model.num_classes=4", "model.params.depth=2", "model.params.width=8",
        "data.img_size=32", "data.batch_size=4", "data.num_workers=0",
        "train.epochs=2", "train.warmup_epochs=1", "train.progress=false",
    ]


def test_cli_train_evaluate_predict(fake_cub, capsys):
    config = str(PROJECT_ROOT / "configs" / "cls_scratch.yaml")
    outputs = fake_cub / "outputs"

    main(["train", "--config", config, *_overrides(fake_cub)])
    (train_dir,) = outputs.glob("cls_scratch_2*")
    best, last = train_dir / "best.pt", train_dir / "last.pt"
    assert best.is_file() and last.is_file()
    assert torch.load(best, weights_only=False)["extra"]["class_names"] == CLASSES

    main(["evaluate", "--config", config, "--ckpt", str(best), "--split", "test", *_overrides(fake_cub)])
    (eval_dir,) = outputs.glob("cls_scratch_eval-test_*")
    with open(eval_dir / "predictions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and rows[0]["true_name"] in CLASSES
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    assert metrics["num_images"] == 12 and 0 <= metrics["f1_macro"] <= 1
    assert (eval_dir / "confusion_matrix.png").is_file()
    assert (eval_dir / "per_class.csv").is_file()

    capsys.readouterr()
    main(["predict", "--config", config, "--ckpt", str(best), "--image", str(fake_cub / "bird.jpg"),
          *_overrides(fake_cub)])
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["prediction"] in CLASSES and len(prediction["top5"]) == 4

    main(["train", "--config", config, "--resume", str(last), *_overrides(fake_cub), "train.epochs=3"])
    resumed_dir = max(outputs.glob("cls_scratch_2*"), key=lambda p: p.stat().st_mtime)
    epochs = [json.loads(line)["epoch"] for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()]
    assert epochs == [1, 2, 3]       # earlier epochs re-logged, then epoch 3 trained
