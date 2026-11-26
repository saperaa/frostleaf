"""Train and score both baselines on the validation split (spec 3.3).

Owner: Tech Lead.

* Trivial baseline: always predict the majority training class.
* Simple learned baseline: logistic regression on images downsampled to
  32 x 32 x 3 and flattened (3,072 features), standardised on train only,
  with balanced class weights.

Both are scored on VALIDATION only; the test split is not touched until the
results are frozen (Phase 4). Results go to experiments/baselines/ and, once
experiments/log.csv exists, one row per baseline is appended to it.

Usage (after the data pipeline in src/data.py is merged):
    python -m src.baselines --author mohamed
    python -m src.baselines --author mohamed --limit 2000   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time

import numpy as np

from src.config import PROJECT_ROOT
from src.evaluate import evaluate, plot_confusion_matrix
from src.models import full_width_proba, logistic_regression_baseline, majority_class_baseline

BASELINE_SIZE = 32  # pixels per side for the logistic-regression input
OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "baselines")
LOG_CSV = os.path.join(PROJECT_ROOT, "experiments", "log.csv")


def dataset_to_arrays(dataset, size: int = BASELINE_SIZE, limit: int | None = None):
    """Turn a dataset of (image, label) pairs into flat feature and label arrays.

    Accepts batched or unbatched ``tf.data.Dataset``s. Images are resized to
    ``size`` x ``size``, scaled to [0, 1] if they are not already, and flattened.

    Args:
        dataset: yields (image, label); image HxWx3 (or NxHxWx3 when batched).
        size: output side length in pixels.
        limit: keep only the first ``limit`` samples (for quick runs).

    Returns:
        (features of shape (N, size*size*3) as float32, labels of shape (N,) as int64).
    """
    import tensorflow as tf

    if len(dataset.element_spec[0].shape) == 4:
        dataset = dataset.unbatch()
    if limit is not None:
        dataset = dataset.take(limit)

    def shrink(image, label):
        image = tf.image.resize(tf.cast(image, tf.float32), (size, size), antialias=True)
        image = tf.cond(tf.reduce_max(image) > 1.5, lambda: image / 255.0, lambda: image)
        return tf.reshape(image, [-1]), label

    feats, labels = [], []
    for x, y in dataset.map(shrink, num_parallel_calls=tf.data.AUTOTUNE).batch(512):
        feats.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)


def _append_log_row(row: dict, log_csv: str | None = LOG_CSV) -> bool:
    """Append ``row`` to the experiment log, matching its header. Skip if absent."""
    if log_csv is None or not os.path.exists(log_csv):
        print(f"note: no experiment log at {log_csv}; result kept in the output folder only")
        return False
    with open(log_csv, newline="") as f:
        header = next(csv.reader(f))
    with open(log_csv, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
    return True


def run_baselines(train_x, train_y, val_x, val_y, class_names: list[str], author: str,
                  out_dir: str = OUT_DIR, log_csv: str | None = LOG_CSV) -> dict:
    """Fit both baselines on train, score them on validation, save and log results.

    Returns:
        {baseline name: summary dict with macro_f1, accuracy, train_time_s, ...}.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_classes = len(class_names)
    date = dt.date.today().isoformat()
    summary = {}

    for name in ("majority_class", "logistic_regression"):
        start = time.perf_counter()
        if name == "majority_class":
            model = majority_class_baseline(train_y)
            params = 0
        else:
            model = logistic_regression_baseline(train_x, train_y)
            lr = model[-1]
            params = int(lr.coef_.size + lr.intercept_.size)
        train_time = time.perf_counter() - start

        train_m = evaluate(train_y, full_width_proba(model, train_x, num_classes), class_names)
        val_m = evaluate(val_y, full_width_proba(model, val_x, num_classes), class_names)
        plot_confusion_matrix(
            val_m["confusion_matrix"], class_names, os.path.join(out_dir, f"{name}_val_cm.png")
        )

        summary[name] = {
            "val_macro_f1": float(val_m["macro_f1"]),
            "val_accuracy": float(val_m["accuracy"]),
            "train_macro_f1": float(train_m["macro_f1"]),
            "train_time_s": round(train_time, 1),
            "params": params,
            "n_train": int(train_y.size),
            "n_val": int(val_y.size),
            "val_roc_auc_macro": float(val_m["roc_auc_macro"]),
            "val_per_class_recall": {c: m["recall"] for c, m in val_m["per_class"].items()},
        }
        _append_log_row({
            "run_id": f"baseline_{name}_{date}",
            "date": date,
            "author": author,
            "model": f"baseline_{name}",
            "precision": "fp32",
            "class_weights": "balanced" if name == "logistic_regression" else "none",
            "seed": 42,
            "train_time_s": summary[name]["train_time_s"],
            "params": params,
            "train_macro_f1": round(summary[name]["train_macro_f1"], 4),
            "val_macro_f1": round(summary[name]["val_macro_f1"], 4),
            "notes": f"{BASELINE_SIZE}x{BASELINE_SIZE} pixels" if params else "",
        }, log_csv)
        print(f"{name:<20} val macro-F1 {summary[name]['val_macro_f1']:.4f}  "
              f"val acc {summary[name]['val_accuracy']:.4f}  ({train_time:.0f}s)")

    with open(os.path.join(out_dir, "baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    """Load the leak-free splits, then fit and score both baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", required=True)
    parser.add_argument("--variant", default="color")
    parser.add_argument("--limit", type=int, default=None, help="cap samples per split")
    args = parser.parse_args()

    from src.data import load_plantvillage

    train_ds, val_ds, _test_ds, class_names = load_plantvillage(args.variant)  # test unused
    train_x, train_y = dataset_to_arrays(train_ds, limit=args.limit)
    val_x, val_y = dataset_to_arrays(val_ds, limit=args.limit)
    run_baselines(train_x, train_y, val_x, val_y, list(class_names), args.author)


if __name__ == "__main__":
    main()
