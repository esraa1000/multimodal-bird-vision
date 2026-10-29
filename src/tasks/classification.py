"""Task 1.2: bird species classification (200 classes).

    python -m src.cli train    --config configs/cls_scratch.yaml
    python -m src.cli evaluate --config configs/cls_scratch.yaml --ckpt <run>/best.pt --split test
    python -m src.cli predict  --config configs/cls_scratch.yaml --ckpt <run>/best.pt --image bird.jpg

``evaluate`` and ``predict`` rebuild the model from the config stored inside
the checkpoint, so they always match the architecture, image size and
normalization the model was trained with.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import Config, deep_merge
from src.data import build_dataloaders
from src.data.transforms import build_transforms
from src.evaluation.metrics import (classification_metrics, confusion_matrix, per_class_metrics,
                                    top_confused_pairs)
from src.models import build_classifier
from src.registry import TASKS
from src.tasks.base import Task
from src.training import Trainer
from src.utils.device import get_device
from src.utils.paths import resolve_path
from src.visualization.plots import plot_confusion_matrix, plot_top_confused_pairs


@TASKS.register("classification")
class ClassificationTask(Task):

    # ------------------------------------------------------------------ train

    def train(self, tracker, run_dir: Path) -> dict:
        cfg = self.cfg
        device = get_device(cfg.get("device", "auto"))
        loaders = build_dataloaders(cfg)
        class_names = list(loaders["train"].dataset.class_names)
        if len(class_names) != cfg.model.num_classes:
            raise ValueError(f"model.num_classes is {cfg.model.num_classes} but the dataset has "
                             f"{len(class_names)} classes")

        init_checkpoint = cfg.train.get("init_checkpoint")
        resume = cfg.train.get("resume")
        model_cfg = copy.deepcopy(cfg.model.to_dict())
        if init_checkpoint or resume:
            # All weights come from the local checkpoint; no ImageNet download.
            if "pretrained" in (model_cfg.get("params") or {}):
                model_cfg["params"]["pretrained"] = False
        model = build_classifier(model_cfg)
        if init_checkpoint and not resume:
            source = resolve_path(init_checkpoint)
            state = torch.load(source, map_location="cpu", weights_only=False)
            saved_names = state.get("extra", {}).get("class_names")
            if saved_names is not None and list(saved_names) != class_names:
                raise ValueError("Initial checkpoint species-label mapping differs from the dataset")
            model.load_state_dict(state["model_state"], strict=True)
            print(f"[classification] initialized model weights from {source} "
                  f"(source epoch {state.get('epoch', '?')}); optimizer and schedule start fresh")
        trainer = Trainer(model, loaders["train"], loaders["val"], cfg, tracker, run_dir, device,
                          checkpoint_extra={"class_names": class_names})
        return trainer.fit(resume_from=resume)

    # --------------------------------------------------------------- evaluate

    def evaluate(self, checkpoint: Path, split: str, tracker, run_dir: Path) -> dict:
        model, ckpt_cfg, class_names, device = self._load(checkpoint)
        # Data comes from the current config, preprocessing from the training config.
        trained_data = ckpt_cfg.get("data", {})
        cfg = Config(deep_merge(self.cfg.to_dict(), {"data": {
            "img_size": trained_data.get("img_size", self.cfg.data.get("img_size", 224)),
            "normalize": trained_data.get("normalize", self.cfg.data.get("normalize", "dataset")),
        }}))
        loader = build_dataloaders(cfg)[split]

        probs, labels = self._predict_loader(model, loader, device)
        preds = probs.argmax(axis=1)
        metrics = classification_metrics(labels, probs)
        cm = confusion_matrix(labels, preds, len(class_names))
        pairs = top_confused_pairs(cm, class_names, k=15)

        run_dir = Path(run_dir)
        file_names = list(getattr(loader.dataset, "file_names", range(len(labels))))
        predictions = self._write_predictions(run_dir / "predictions.csv", file_names, labels, probs, class_names)
        per_class = self._write_csv(run_dir / "per_class.csv", per_class_metrics(labels, preds, class_names))
        np.save(run_dir / "confusion_matrix.npy", cm)
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"split": split, "checkpoint": str(checkpoint), **metrics,
                       "top_confused_pairs": pairs}, f, indent=2)

        result = {f"{split}/{k}": v for k, v in metrics.items()}
        tracker.log_metrics(result)
        tracker.log_summary(result)
        tracker.log_artifact(predictions, name=f"predictions_{split}", kind="predictions")
        tracker.log_artifact(per_class, name=f"per_class_{split}", kind="evaluation")
        tracker.log_figure(plot_confusion_matrix(cm, run_dir / "confusion_matrix.png",
                                                 title=f"Confusion matrix ({split})"), name="confusion_matrix")
        if pairs:
            tracker.log_figure(plot_top_confused_pairs(pairs, run_dir / "top_confused.png",
                                                       title=f"Most confused classes ({split})"),
                               name="top_confused")
        return result

    # ---------------------------------------------------------------- predict

    def predict(self, checkpoint: Path, image_path: Path, **options) -> dict:
        model, ckpt_cfg, class_names, device = self._load(checkpoint)
        data = ckpt_cfg.get("data", {})
        transform = build_transforms(False, data.get("img_size", 224), "none", data.get("normalize", "dataset"))
        x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = model(x).softmax(dim=1)[0].cpu()
        conf, idx = probs.topk(min(5, len(class_names)))
        top5 = [{"class_id": int(i), "name": class_names[i], "confidence": round(float(c), 4)}
                for c, i in zip(conf, idx)]
        result = {"image": str(image_path), "prediction": top5[0]["name"],
                  "confidence": top5[0]["confidence"], "top5": top5}
        if options.get("gradcam"):
            result["gradcam"] = "not implemented yet (planned: src/visualization/gradcam.py)"
        return result

    # ---------------------------------------------------------------- helpers

    def _load(self, checkpoint: Path):
        """Rebuild the model exactly as it was trained and load its weights."""
        device = get_device(self.cfg.get("device", "auto"))
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        ckpt_cfg = state.get("config") or self.cfg.to_dict()

        model_cfg = copy.deepcopy(ckpt_cfg["model"])
        model_cfg["freeze_backbone"] = False
        if "pretrained" in (model_cfg.get("params") or {}):
            model_cfg["params"]["pretrained"] = False   # weights come from the checkpoint
        model = build_classifier(model_cfg)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()

        class_names = state.get("extra", {}).get("class_names") or [
            str(i) for i in range(model_cfg["num_classes"])]
        return model, ckpt_cfg, class_names, device

    @staticmethod
    def _predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
        all_probs, all_labels = [], []
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for images, labels in loader:
                logits = model(images.to(device, non_blocking=True))
                all_probs.append(logits.float().softmax(dim=1).cpu())
                all_labels.append(labels)
        return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()

    @staticmethod
    def _write_predictions(path: Path, file_names, labels, probs, class_names) -> Path:
        """One row per image; the input for error analysis and Grad-CAM picks."""
        top5 = np.argsort(-probs, axis=1)[:, :5]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "true_id", "true_name", "pred_id", "pred_name",
                             "confidence", "top5_ids", "correct"])
            for name, y, p, t5 in zip(file_names, labels, probs, top5):
                pred = int(t5[0])
                writer.writerow([name, int(y), class_names[y], pred, class_names[pred],
                                 round(float(p[pred]), 4), " ".join(map(str, t5)), int(pred == y)])
        return path

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> Path:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path
