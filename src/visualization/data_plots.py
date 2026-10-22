"""Data figures for the report (owner: M2).

    python -m src.visualization.data_plots          # -> reports/figures/augmentation_examples.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # save to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.download import load_cub  # noqa: E402
from src.data.transforms import build_transforms, get_mean_std  # noqa: E402
from src.utils.paths import get_data_dir, resolve_path  # noqa: E402


def plot_augmentation_examples(images: list, out_path: Path, img_size: int = 224,
                               normalize: str = "dataset", draws: int = 2, seed: int = 0) -> Path:
    """One row per augmentation level (none / light / strong), ``draws`` random results per image."""
    torch.manual_seed(seed)   # same figure every run
    mean, std = (torch.tensor(v).view(3, 1, 1) for v in get_mean_std(normalize))
    levels = ["none", "light", "strong"]

    fig, axes = plt.subplots(len(levels), len(images) * draws, figsize=(2.5 * len(images) * draws, 8))
    for r, level in enumerate(levels):
        tf = build_transforms(level != "none", img_size, level, normalize)
        for b, img in enumerate(images):
            for k in range(draws):
                x = (tf(img) * std + mean).clamp(0, 1)   # undo normalization for display
                ax = axes[r, b * draws + k]
                ax.imshow(x.permute(1, 2, 0))
                ax.axis("off")
        axes[r, 0].set_title(level, loc="left", fontsize=12)
    fig.suptitle(f"Augmentation examples: none / light / strong ({draws} random draws per bird)")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the augmentation examples figure.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="reports/figures/augmentation_examples.png")
    parser.add_argument("--n-birds", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(resolve_path(cfg.data.get("split_file", "configs/data/split_seed42.json")), encoding="utf-8") as f:
        train_names = json.load(f)["train"]          # only training images, never val/test

    train_hf = load_cub(get_data_dir(cfg.get("paths", {})), "train")
    row_of = {name: row for row, name in enumerate(train_hf["file_name"])}
    step = len(train_names) // args.n_birds          # spread the birds across species
    picks = [train_names[i * step] for i in range(args.n_birds)]
    images = [train_hf[row_of[name]]["image"].convert("RGB") for name in picks]

    out = plot_augmentation_examples(images, resolve_path(args.out), cfg.data.get("img_size", 224))
    print(f"saved {out} ({', '.join(picks)})")


if __name__ == "__main__":
    main()
