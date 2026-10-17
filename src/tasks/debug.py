"""A tiny end-to-end task on random data, used to test the infrastructure.

It exercises config -> seed -> model -> training loop -> tracker -> checkpoint
-> evaluate -> predict without downloading the dataset, so anyone can check
their environment in a few seconds::

    python -m src.cli train --task debug --config configs/debug.yaml
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.registry import TASKS
from src.tasks.base import Task
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_device


def _tiny_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(8, num_classes),
    )


def _random_batch(n: int, size: int, num_classes: int, generator: torch.Generator):
    x = torch.randn(n, 3, size, size, generator=generator)
    y = torch.randint(0, num_classes, (n,), generator=generator)
    return x, y


@TASKS.register("debug")
class DebugTask(Task):
    def train(self, tracker, run_dir: Path) -> dict:
        c = self.cfg
        device = get_device(c.get("device", "auto"))
        num_classes = c.model.num_classes
        model = _tiny_model(num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.train.lr)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(c.seed)

        best_acc, metrics = -1.0, {}
        for epoch in range(1, c.train.epochs + 1):
            model.train()
            x, y = _random_batch(c.data.batch_size, c.data.img_size, num_classes, gen)
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                xv, yv = _random_batch(c.data.batch_size, c.data.img_size, num_classes, gen)
                acc = (model(xv.to(device)).argmax(1).cpu() == yv).float().mean().item()

            metrics = {"train/loss": loss.item(), "val/acc": acc, "epoch": epoch}
            tracker.log_metrics(metrics, step=epoch)
            if acc > best_acc:
                best_acc = acc
                ckpt = save_checkpoint(run_dir / "best.pt", model, optimizer, epoch=epoch,
                                       metrics=metrics, config=c.to_dict())

        tracker.log_artifact(ckpt, name="best_model", kind="model")
        tracker.log_summary({"best/val_acc": best_acc})
        return {"best_val_acc": best_acc, "checkpoint": str(ckpt)}

    def evaluate(self, checkpoint: Path, split: str, tracker, run_dir: Path) -> dict:
        model = _tiny_model(self.cfg.model.num_classes)
        load_checkpoint(checkpoint, model)
        model.eval()
        gen = torch.Generator().manual_seed(self.cfg.seed + 1)
        x, y = _random_batch(64, self.cfg.data.img_size, self.cfg.model.num_classes, gen)
        with torch.no_grad():
            acc = (model(x).argmax(1) == y).float().mean().item()
        result = {f"{split}/acc": acc}
        tracker.log_metrics(result)
        tracker.log_summary(result)
        return result

    def predict(self, checkpoint: Path, image_path: Path, **options) -> dict:
        model = _tiny_model(self.cfg.model.num_classes)
        load_checkpoint(checkpoint, model)
        model.eval()
        x = torch.randn(1, 3, self.cfg.data.img_size, self.cfg.data.img_size)  # ignores the image
        with torch.no_grad():
            probs = model(x).softmax(1)[0]
        conf, idx = probs.max(0)
        return {"image": str(image_path), "class_id": int(idx), "confidence": round(float(conf), 4)}
