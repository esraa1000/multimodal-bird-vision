"""Generic supervised training loop, shared by every classification model.

    trainer = Trainer(model, loaders["train"], loaders["val"], cfg, tracker, run_dir, device)
    result = trainer.fit()                          # or fit(resume_from=".../last.pt")

Each epoch: train -> validate -> log metrics -> save ``last.pt`` (to resume an
interrupted run) and ``best.pt`` (when val accuracy improves) -> early-stopping
check. At the end: training-curves figure, summary values, and the best
checkpoint uploaded to the tracker once.

The trainer knows nothing about which model, dataset or tracker it is given,
so swapping the encoder never touches this file.

Config keys (``train`` section): epochs, lr, weight_decay, optimizer, scheduler,
warmup_epochs, min_lr, label_smoothing, amp, grad_clip, early_stopping_patience,
max_batches, upload_checkpoint, progress. ``model.backbone_lr_mult`` gives a
pretrained encoder a smaller learning rate.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from src.tracking import Tracker
from src.training.optim import build_optimizer, build_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.visualization.plots import plot_training_curves


class Trainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, cfg, tracker: Tracker,
                 run_dir: str | Path, device: torch.device, loss_fn: nn.Module | None = None,
                 checkpoint_extra: dict | None = None):
        """
        Args:
            cfg: the full run config (stored inside every checkpoint).
            checkpoint_extra: saved in each checkpoint, e.g. ``{"class_names": [...]}``
                so inference can print species names.
        """
        self.cfg = cfg
        self.t = cfg.train
        self.device = device
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.tracker = tracker
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(label_smoothing=float(self.t.get("label_smoothing", 0.0)))
        self.checkpoint_extra = checkpoint_extra or {}

        self.epochs = int(self.t.epochs)
        self.optimizer = build_optimizer(self.model, self.t, cfg.get_path("model.backbone_lr_mult", 1.0))
        self.scheduler = build_scheduler(self.optimizer, self.t, self.epochs)
        self.use_amp = bool(self.t.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = self.t.get("grad_clip")
        self.max_batches = self.t.get("max_batches")

        self.history: list[dict] = []
        self.best_acc = -1.0
        self.best_epoch = 0
        self.bad_epochs = 0          # epochs since the last improvement (early stopping)
        self.start_epoch = 1

    # ------------------------------------------------------------------ public

    def fit(self, resume_from: str | Path | None = None) -> dict:
        if resume_from:
            self._resume(Path(resume_from))

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[trainer] {n_params / 1e6:.2f}M parameters ({n_trainable / 1e6:.2f}M trainable) | "
              f"device={self.device} amp={self.use_amp} | epochs {self.start_epoch}..{self.epochs}")

        patience = self.t.get("early_stopping_patience")
        stopped_early = False
        t_start = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            lr = self.optimizer.param_groups[-1]["lr"]   # the head's lr (the last group)
            train_stats = self._run_epoch(self.train_loader, train=True, desc=f"epoch {epoch}/{self.epochs} train")
            val_stats = self._run_epoch(self.val_loader, train=False, desc=f"epoch {epoch}/{self.epochs} val")
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {
                **{f"train/{k}": v for k, v in train_stats.items()},
                **{f"val/{k}": v for k, v in val_stats.items()},
                "lr": lr,
                "epoch": epoch,
                "time/epoch_s": round(time.time() - t0, 1),
            }
            self.tracker.log_metrics(metrics, step=epoch)
            self.history.append(metrics)

            if val_stats["acc"] > self.best_acc:
                self.best_acc, self.best_epoch, self.bad_epochs = val_stats["acc"], epoch, 0
                self._save(self.run_dir / "best.pt", epoch, metrics, for_resume=False)
            else:
                self.bad_epochs += 1
            self._save(self.run_dir / "last.pt", epoch, metrics, for_resume=True)

            if patience and self.bad_epochs >= patience:
                print(f"[trainer] early stopping: val/acc has not improved for {patience} epochs")
                stopped_early = True
                break

        return self._finish(stopped_early, time.time() - t_start, n_params)

    # ----------------------------------------------------------------- one epoch

    def _run_epoch(self, loader, train: bool, desc: str) -> dict:
        self.model.train(train)
        total_loss, correct, correct_top5, seen = 0.0, 0, 0, 0
        bar = tqdm(loader, desc=desc, leave=False, disable=not self.t.get("progress", True))

        with torch.set_grad_enabled(train):
            for i, (images, labels) in enumerate(bar):
                if self.max_batches and i >= self.max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss became {loss.item()} ({desc}). Try a lower learning rate.")
                if train:
                    self._step(loss)

                n = labels.size(0)
                k = min(5, logits.size(1))
                top_k = logits.topk(k, dim=1).indices
                total_loss += loss.item() * n
                correct += (top_k[:, 0] == labels).sum().item()
                correct_top5 += (top_k == labels[:, None]).any(dim=1).sum().item()
                seen += n
                bar.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

        if seen == 0:
            raise RuntimeError(f"No batches were run ({desc}): the data loader is empty.")
        return {"loss": total_loss / seen, "acc": correct / seen, "top5_acc": correct_top5 / seen}

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)     # clip the real gradients, not the scaled ones
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # --------------------------------------------------------------- checkpoints

    def _save(self, path: Path, epoch: int, metrics: dict, for_resume: bool) -> None:
        """``last.pt`` keeps optimizer/scheduler state to resume; ``best.pt`` keeps
        only what inference needs, so it is about 3x smaller to upload."""
        extra = {**self.checkpoint_extra, "best_val_acc": self.best_acc, "best_epoch": self.best_epoch}
        if for_resume:
            extra.update(bad_epochs=self.bad_epochs, history=self.history,
                         scaler_state=self.scaler.state_dict())
        save_checkpoint(
            path, self.model,
            optimizer=self.optimizer if for_resume else None,
            scheduler=self.scheduler if for_resume else None,
            epoch=epoch, metrics=metrics, config=self.cfg.to_dict(), extra=extra,
        )

    def _resume(self, path: Path) -> None:
        state = load_checkpoint(path, self.model, self.optimizer, self.scheduler, map_location=self.device)
        extra = state.get("extra", {})
        self.start_epoch = int(state["epoch"]) + 1
        self.best_acc = extra.get("best_val_acc", -1.0)
        self.best_epoch = extra.get("best_epoch", 0)
        self.bad_epochs = extra.get("bad_epochs", 0)
        self.history = list(extra.get("history", []))
        if extra.get("scaler_state"):
            self.scaler.load_state_dict(extra["scaler_state"])

        # Keep the best model found before the interruption in the new run folder.
        old_best = path.parent / "best.pt"
        new_best = self.run_dir / "best.pt"
        if old_best.is_file() and old_best.resolve() != new_best.resolve():
            shutil.copy2(old_best, new_best)
        # Re-log the earlier epochs so the new run shows the complete curves.
        for metrics in self.history:
            self.tracker.log_metrics(metrics, step=metrics["epoch"])
        print(f"[trainer] resumed from {path}: continuing at epoch {self.start_epoch} "
              f"(best val/acc {self.best_acc:.4f} at epoch {self.best_epoch})")

    # ------------------------------------------------------------------- the end

    def _finish(self, stopped_early: bool, elapsed_s: float, n_params: int) -> dict:
        summary = {
            "best/val_acc": self.best_acc,
            "best/epoch": self.best_epoch,
            "epochs_trained": self.history[-1]["epoch"] if self.history else 0,
            "stopped_early": stopped_early,
            "model/params": n_params,
            "time/train_s": round(elapsed_s, 1),
        }
        self.tracker.log_summary(summary)

        if self.history:
            curves = plot_training_curves(self.history, self.run_dir / "training_curves.png")
            self.tracker.log_figure(curves, name="training_curves")

        best = self.run_dir / "best.pt"
        if best.is_file() and self.t.get("upload_checkpoint", True):
            self.tracker.log_artifact(best, name="best_model", kind="model")
        return {**summary, "checkpoint": str(best), "last_checkpoint": str(self.run_dir / "last.pt")}
