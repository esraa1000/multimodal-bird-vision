"""Training and evaluation figures (curves, confusion matrix, most-confused classes)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # save to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(history: list[dict], out_path: str | Path) -> Path:
    """Train vs. validation loss and accuracy per epoch (from ``Trainer.history``)."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        ax_loss.plot(epochs, [h[f"{split}/loss"] for h in history], marker=".", label=split)
        ax_acc.plot(epochs, [h[f"{split}/acc"] for h in history], marker=".", label=split)
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax_acc.set(title="Top-1 accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, out_path: str | Path, title: str = "Confusion matrix") -> Path:
    """Row-normalized confusion matrix. With 200 classes the labels are class ids:
    a bright diagonal is good, off-diagonal spots show which species get mixed up."""
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="fraction of the true class")
    ax.set(title=title, xlabel="predicted class id", ylabel="true class id")
    return _save(fig, out_path)


def plot_top_confused_pairs(pairs: list[dict], out_path: str | Path, title: str = "Most confused classes") -> Path:
    """Bar chart of the ``true -> predicted`` pairs with the most mistakes."""
    labels = [f"{p['true_name']}  ->  {p['pred_name']}" for p in pairs][::-1]
    counts = [p["count"] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.4 * max(len(pairs), 1) + 1))
    ax.barh(labels, counts)
    ax.set(title=title, xlabel="number of misclassified images")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_path)
