"""Compare selected checkpoints on six labeled validation images, without training.

Run from the repository root; see docs/gradcam_analysis.md for checkpoint sources.
All heatmaps are generated through the same predict CLI used interactively.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.cli import main as cli
from src.config import load_config
from src.data.dataset import CUBDataset
from src.data.download import load_cub
from src.data.transforms import build_transforms
from src.tasks.classification import ClassificationTask
from src.utils.paths import resolve_path
from src.visualization.gradcam import compute_gradcam
from src.visualization.plots import _save, plt

CONFIGS = {"scratch": "cls_scratch_residual.yaml", "resnet50": "cls_resnet50.yaml",
           "efficientnet": "cls_efficientnet.yaml"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def display_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", default="outputs/gradcam_checkpoints/scratch/best.pt")
    parser.add_argument("--resnet50", default="outputs/cls_resnet50_frozen_20261029-034949/best.pt")
    parser.add_argument("--efficientnet", default="outputs/gradcam_checkpoints/efficientnet/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="reports/gradcam")
    args = parser.parse_args()
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    raw_output = resolve_path("outputs/gradcam_analysis")
    split = json.loads((ROOT / "configs/data/split_seed42.json").read_text())
    # Alphabetical filenames and distinct true species make selection deterministic.
    names = sorted(split["val"])
    hf = load_cub(resolve_path(args.data_dir), "train")
    class_names = hf.features["label"].names
    row_of = {name: i for i, name in enumerate(hf["file_name"])}
    labels = np.array([hf[row_of[name]]["label"] for name in names])
    predictions, verification = {}, {}
    for key, config in CONFIGS.items():
        checkpoint = resolve_path(getattr(args, key))
        before_hash = sha256(checkpoint)
        cfg = load_config(ROOT / "configs" / config, [f"device={args.device}"])
        task = ClassificationTask(cfg)
        model, trained, saved_names, device = task._load(checkpoint)
        if saved_names != class_names:
            raise ValueError(f"Species mapping mismatch for {key}")
        d = trained["data"]
        ds = CUBDataset(hf, names, build_transforms(False, d["img_size"], "none", d["normalize"]))
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for images, _ in loader:
                # FP32 matches the single-image predict CLI; no test-set selection.
                probs.append(model(images.to(device)).softmax(dim=1).cpu())
        predictions[key] = torch.cat(probs).numpy()
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        flags = [p.requires_grad for p in model.parameters()]
        x = ds[0][0].unsqueeze(0).to(device)
        with torch.no_grad():
            original_logits = model(x)
        explanation = compute_gradcam(model, x)
        torch.testing.assert_close(explanation["logits"], original_logits)
        unchanged = all(torch.equal(tensor.detach().cpu(), state[name])
                        for name, tensor in model.state_dict().items())
        assert unchanged and flags == [p.requires_grad for p in model.parameters()]
        assert all(p.grad is None for p in model.parameters())
        assert not model.training
        verification[key] = {"checkpoint": display_path(checkpoint),
            "sha256_before": before_hash, "parameters_and_buffers_unchanged": unchanged,
            "parameter_gradients_untouched": True, "prediction_matches_plain_forward": True,
            "target_layer": explanation["layer"], "feature_map_size": explanation["feature_map_size"],
            "model_config": trained["model"],
            "preprocessing": {"img_size": d["img_size"], "normalize": d["normalize"]}}
        print(f"{key}: checked {len(ds)} validation images; state unchanged; layer={explanation['layer']}", flush=True)
        del model, state, probs, explanation

    correct = {key: probs.argmax(axis=1) == labels for key, probs in predictions.items()}
    all_correct = np.logical_and.reduce(list(correct.values()))
    all_wrong = ~np.logical_or.reduce(list(correct.values()))
    selections, used_labels = [], set()

    def pick(mask, group, count):
        selected = 0
        for i in np.flatnonzero(mask):
            if int(labels[i]) in used_labels:
                continue
            selections.append((int(i), group))
            used_labels.add(int(labels[i]))
            selected += 1
            if selected == count:
                return
        raise RuntimeError(f"Not enough distinct validation species for {group}")

    pick(all_correct, "all_correct", 2)
    pick(all_wrong, "all_incorrect", 2)
    pick(~correct["scratch"] & correct["efficientnet"], "scratch_wrong_efficientnet_correct", 1)
    pick(correct["resnet50"] & ~correct["efficientnet"], "resnet_correct_efficientnet_wrong", 1)
    cases = []
    for n, (index, group) in enumerate(selections, 1):
        case_id = f"case{n:02d}"
        name, label = names[index], int(labels[index])
        image = hf[row_of[name]]["image"].convert("RGB")
        image_path = image_dir / f"{case_id}.png"
        image.save(image_path)  # lossless export: CLI sees the exact same decoded pixels
        case = {"id": case_id, "selection_group": group, "file_name": name,
                "label_id": label, "label": class_names[label],
                "image": display_path(image_path), "models": {}}
        fig, axes = plt.subplots(1, 4, figsize=(13, 4))
        for col, (key, config) in enumerate(CONFIGS.items(), 1):
            checkpoint = resolve_path(getattr(args, key))
            base = ["predict", "--config", str(ROOT / "configs" / config),
                    "--ckpt", str(checkpoint), "--image", str(image_path), "--gradcam",
                    "--label", str(label), "--output-dir", str(raw_output / case_id / key)]
            stream = io.StringIO()
            with redirect_stdout(stream):
                cli(base + [f"device={args.device}"])
            result = json.loads(stream.getvalue())
            assert result["prediction_id"] == int(predictions[key][index].argmax())
            gc = result["gradcam"]
            with Image.open(gc["overlay"]) as overlay:
                axes[col].imshow(overlay.copy())
            if col == 1:
                with Image.open(gc["input"]) as crop:
                    axes[0].imshow(crop.copy())
                axes[0].set_title("Model input crop")
            status = "Correct" if result["correct"] else "Incorrect"
            axes[col].set_title(f"{key}: {status}\n{result['prediction']}\n{result['confidence']:.1%}", fontsize=9)
            model_result = {k: result[k] for k in ("prediction_id", "prediction", "confidence", "correct")}
            model_result.update(target_layer=gc["layer"], positive_evidence=gc["positive_evidence"],
                                predicted_class_metadata=display_path(Path(gc["metadata"])))
            if not result["correct"]:
                stream = io.StringIO()
                with redirect_stdout(stream):
                    cli(base + ["--gradcam-class", str(label), f"device={args.device}"])
                true_map = json.loads(stream.getvalue())
                model_result["true_class_metadata"] = display_path(Path(true_map["gradcam"]["metadata"]))
                # Retain the true-class contrast figure alongside the shared comparison.
                with Image.open(true_map["gradcam"]["comparison"]) as comparison:
                    comparison.save(output / f"{case_id}_{key}_true_class.png")
            case["models"][key] = model_result
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{case_id}: {name}\nDataset label: {class_names[label]} | {group}", fontsize=11)
        _save(fig, output / f"{case_id}.png", rect=(0, 0, 1, 0.84))
        cases.append(case)
        print(f"Saved {case_id}: {name} ({group})", flush=True)
    for key in CONFIGS:
        after_hash = sha256(resolve_path(getattr(args, key)))
        verification[key]["sha256_after"] = after_hash
        assert after_hash == verification[key]["sha256_before"]
    record = {"split": "validation", "num_candidates": len(names),
              "selection_rule": "First alphabetically named image from each distinct species satisfying each correctness group; no selection by heatmap appearance.",
              "validation_correct_counts_fp32": {key: int(mask.sum()) for key, mask in correct.items()},
              "verification": verification, "cases": cases}
    (output / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("All three checkpoint SHA-256 hashes unchanged. No training performed.", flush=True)


if __name__ == "__main__":
    main()
