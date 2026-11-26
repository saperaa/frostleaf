"""Metrics and error analysis: every number in the project is computed here.

Owner: Evaluation & Deployment Engineer.

`evaluate()` scores one set of predictions: accuracy, macro-F1 (primary
metric), per-class precision/recall/F1/support, the confusion matrix and
one-vs-rest ROC-AUC per class plus its macro average. It takes labels and
probabilities rather than a model, so baselines (scikit-learn) and Keras
models are scored by exactly the same code.

Classes absent from the evaluated set (common on PlantDoc, which covers only
some PlantVillage classes) are handled explicitly:
* macro-F1 averages over the classes that occur in `y_true`, so a class the
  set cannot test does not count as a zero;
* ROC-AUC is skipped for a class with no positive (or no negative) sample,
  and the skipped classes are listed in the result.

Week 13: `worst_predictions` for error analysis (>= 20 misclassified
samples, failure categories, Grad-CAM).
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _check_inputs(y_true, y_prob, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes and label range; return (y_true, y_prob) as numpy arrays.

    Args:
        y_true: integer class indices, shape (N,).
        y_prob: predicted probabilities, shape (N, K).
        class_names: K class names, index i names column i of `y_prob`.

    Raises:
        ValueError: on empty input, mismatched shapes or out-of-range labels.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_classes = len(class_names)
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if y_prob.ndim != 2 or y_prob.shape != (y_true.size, n_classes):
        raise ValueError(
            f"y_prob must have shape (N, K) = ({y_true.size}, {n_classes}), got {y_prob.shape}"
        )
    if y_true.min() < 0 or y_true.max() >= n_classes:
        raise ValueError(f"labels must be in [0, {n_classes - 1}]")
    return y_true, y_prob


def evaluate(y_true, y_prob, class_names: list[str]) -> dict:
    """Compute the full metric set for one set of predictions.

    Args:
        y_true: true class indices, shape (N,).
        y_prob: predicted class probabilities, shape (N, K) with K =
            len(class_names) (38 for PlantVillage). The predicted class is the
            argmax of each row.
        class_names: names of the K classes, in column order.

    Returns:
        A dictionary with
        * ``accuracy`` and ``macro_f1`` (primary metric; mean F1 over the
          classes present in ``y_true``),
        * ``per_class``: ``{name: {precision, recall, f1, support, roc_auc}}``
          for all K classes (recall, F1 and AUC are NaN for absent classes),
        * ``confusion_matrix``: K x K counts, rows = true, columns = predicted,
        * ``roc_auc_per_class``: one-vs-rest AUC for every class it is
          defined for, and ``roc_auc_macro``: their mean (NaN if none),
        * ``auc_skipped_classes``: classes whose AUC is undefined because the
          set has no positive or no negative sample for them,
        * ``absent_classes``: classes with zero support, ``n_samples``.
    """
    y_true, y_prob = _check_inputs(y_true, y_prob, class_names)
    n_classes = len(class_names)
    all_labels = np.arange(n_classes)
    y_pred = y_prob.argmax(axis=1)

    support = np.bincount(y_true, minlength=n_classes)
    present = all_labels[support > 0]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )
    recall = np.where(support > 0, recall, np.nan)
    f1 = np.where(support > 0, f1, np.nan)

    auc = np.full(n_classes, np.nan)
    skipped = []
    for k in all_labels:
        positives = support[k]
        if positives == 0 or positives == y_true.size:
            skipped.append(class_names[k])
            continue
        auc[k] = roc_auc_score((y_true == k).astype(int), y_prob[:, k])

    per_class = {
        name: {
            "precision": float(precision[k]),
            "recall": float(recall[k]),
            "f1": float(f1[k]),
            "support": int(support[k]),
            "roc_auc": float(auc[k]),
        }
        for k, name in enumerate(class_names)
    }
    computed_auc = auc[~np.isnan(auc)]

    return {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1[present])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels),
        "roc_auc_per_class": {
            class_names[k]: float(auc[k]) for k in all_labels if not np.isnan(auc[k])
        },
        "roc_auc_macro": float(computed_auc.mean()) if computed_auc.size else float("nan"),
        "auc_skipped_classes": skipped,
        "absent_classes": [class_names[k] for k in all_labels if support[k] == 0],
    }


def normalize_rows(cm) -> np.ndarray:
    """Divide each confusion-matrix row by its total, so the diagonal is recall.

    Rows with no samples (a class absent from the set) become all-NaN rather
    than dividing by zero.
    """
    cm = np.asarray(cm, dtype=np.float64)
    totals = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, cm / totals, np.nan)


def plot_confusion_matrix(cm, class_names: list[str], path: str) -> str:
    """Save a row-normalised confusion-matrix heatmap (each row shows recall).

    Readable at 38 classes: the figure grows with the class count, every
    class name is printed on both axes, and each non-zero cell carries its
    percentage. Absent classes (empty rows) are drawn grey.

    Args:
        cm: K x K confusion matrix of counts (rows = true, columns = predicted),
            e.g. ``evaluate(...)["confusion_matrix"]``.
        class_names: the K class names, in row/column order.
        path: output image file (PNG recommended); parent folders are created.

    Returns:
        The path written.
    """
    norm = normalize_rows(cm)
    n = len(class_names)
    if norm.shape != (n, n):
        raise ValueError(f"cm must be {n} x {n}, got {norm.shape}")

    size = max(6.0, 0.32 * n + 3.0)
    fig = Figure(figsize=(size, size))
    ax = fig.add_subplot()
    masked = np.ma.masked_invalid(norm)
    cmap = _heatmap_cmap()
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")

    font = 9 if n <= 15 else 6
    ax.set_xticks(np.arange(n), labels=class_names, rotation=90, fontsize=font)
    ax.set_yticks(np.arange(n), labels=class_names, fontsize=font)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (row-normalised: diagonal = recall)")

    for i in range(n):
        for j in range(n):
            value = norm[i, j]
            if np.isnan(value) or value == 0:
                continue
            ax.text(
                j, i, f"{100 * value:.0f}", ha="center", va="center",
                fontsize=font - 1, color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=200)
    return path


def _heatmap_cmap():
    """Return a white-to-blue colormap that draws NaN (absent-class) cells grey."""
    from matplotlib import colormaps

    return colormaps["Blues"].with_extremes(bad="lightgrey")


def worst_predictions(model, dataset, k: int = 20):
    """Return the k most confidently wrong samples for error analysis."""
    raise NotImplementedError("Evaluation Engineer: implement in Week 13")
