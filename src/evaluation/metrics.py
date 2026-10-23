"""Classification metrics. What each one tells you:

* ``acc`` (top-1 accuracy): fraction of images whose most likely class is the
  true species. Easy to read, and fair here because CUB classes are balanced
  (~30 images per class).
* ``top5_acc``: fraction of images whose true species is among the 5 most
  likely. Shows whether a mistake was a near miss, which is common between
  similar-looking species such as the many warblers and sparrows.
* ``precision_macro``: for each class, how many images predicted as that class
  really are it, averaged over classes. Low precision = the model "over-uses"
  some classes.
* ``recall_macro``: for each class, how many of its images were found,
  averaged over classes. On a balanced dataset this is close to accuracy.
* ``f1_macro``: per-class harmonic mean of precision and recall, averaged.
  Every species counts equally, so one class that is never predicted lowers it
  even when overall accuracy looks fine.
* confusion matrix: counts of (true class, predicted class); its off-diagonal
  cells show exactly which species get mixed up.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """``y_true``: (N,) class ids. ``probs``: (N, num_classes) predicted probabilities."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    k = min(5, probs.shape[1])
    top_k = np.argsort(-probs, axis=1)[:, :k]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "acc": float((y_pred == y_true).mean()),
        "top5_acc": float((top_k == y_true[:, None]).any(axis=1).mean()),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "num_images": int(len(y_true)),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of images of class ``i`` predicted as class ``j``."""
    return _confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> list[dict]:
    """Precision / recall / F1 / support per species, for finding the hardest classes."""
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return [
        {"class_id": int(i), "name": class_names[i], "support": int(support[i]),
         "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i in labels
    ]


def top_confused_pairs(cm: np.ndarray, class_names: list[str], k: int = 10) -> list[dict]:
    """The ``k`` (true, predicted) pairs with the most mistakes."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    pairs = []
    for flat in order:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] == 0:
            break
        pairs.append({"true_id": int(i), "true_name": class_names[i],
                      "pred_id": int(j), "pred_name": class_names[j], "count": int(off[i, j])})
    return pairs
