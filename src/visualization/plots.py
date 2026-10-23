"""Training and evaluation figures (curves, confusion matrix, most-confused classes)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # save to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
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


def plot_tuning_comparison(trials_by_method: dict[str, list[dict]], out_path: str | Path,
                           param: str, metric_label: str = "best val accuracy") -> Path:
    """Left: best score found so far vs. number of trials (who finds good values sooner).
    Right: each trial's score against one hyperparameter (which values the search tried)."""
    fig, (ax_best, ax_param) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, trials in trials_by_method.items():
        trials = sorted(trials, key=lambda r: r["trial_id"])
        if not trials:
            continue
        scores = np.array([r["score"] for r in trials])
        x = np.arange(1, len(scores) + 1)
        line, = ax_best.step(x, np.maximum.accumulate(scores), where="post", label=label)
        ax_best.scatter(x, scores, s=14, alpha=0.4, color=line.get_color())
        values = [r["params"].get(param) for r in trials]
        ax_param.scatter(values, scores, s=30, alpha=0.8, color=line.get_color(), label=label)
        best = int(scores.argmax())
        ax_param.scatter([values[best]], [scores[best]], s=160, marker="*", color=line.get_color(),
                         edgecolors="black")

    all_values = [r["params"].get(param) for t in trials_by_method.values() for r in t]
    numeric = [v for v in all_values if isinstance(v, (int, float))]
    if numeric and min(numeric) > 0 and max(numeric) / min(numeric) >= 10:
        ax_param.set_xscale("log")   # e.g. learning rates spanning orders of magnitude
        ax_param.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax_param.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_param.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_best.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_best.set(title="Best score so far", xlabel="trials completed", ylabel=metric_label)
    ax_param.set(title=f"Score vs. {param} (star = best)", xlabel=param, ylabel=metric_label)
    for ax in (ax_best, ax_param):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return _save(fig, out_path)
