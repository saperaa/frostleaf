"""Baselines and neural architectures.

Owners: Tech Lead (baselines) and Model Architect (Models A, B, C).

Every network is built as a `src.precision.PrecisionModel`, so the same
architecture can be trained in fp32, bf16_rtn or bf16_sr by changing the config.

Models (spec 3.4, image track):
* Model A: custom CNN trained from scratch (larger LR, e.g. 1e-3).
* Model B: ResNet-50 (ImageNet weights) frozen + new classifier head.
* Model C: Model B with the last block unfrozen, small LR (e.g. 1e-5).
  Start from Model B's trained weights (head included).
  Keep BatchNorm layers in inference mode (training=False) while fine-tuning.

For each model the report needs: architecture diagram, output shape per layer,
hand-computed parameter counts checked against model.summary(), and a one- or
two-sentence justification for every design choice.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def majority_class_baseline(train_labels) -> DummyClassifier:
    """Trivial baseline: always predict the most frequent training class.

    Args:
        train_labels: integer class labels of the TRAIN split.

    Returns:
        A fitted scikit-learn ``DummyClassifier``. It ignores its input, so any
        2-D array with one row per sample can be passed to ``predict_proba``.
    """
    train_labels = np.asarray(train_labels)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((train_labels.size, 1)), train_labels)
    return model


def logistic_regression_baseline(
    train_features, train_labels, max_iter: int = 300, seed: int = 42
) -> Pipeline:
    """Simple learned baseline: standardise, then multinomial logistic regression.

    The scaler is inside the pipeline, so it is fitted on the training split
    only and re-applied unchanged to validation and test (no leakage).
    ``class_weight="balanced"`` counters class imbalance.

    Args:
        train_features: TRAIN features, shape (N, D), e.g. flattened 32x32x3 pixels.
        train_labels: integer class labels of the TRAIN split, shape (N,).
        max_iter: maximum L-BFGS iterations.
        seed: random seed.

    Returns:
        The fitted scikit-learn ``Pipeline``.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=seed),
    )
    model.fit(np.asarray(train_features), np.asarray(train_labels))
    return model


def full_width_proba(model, features, num_classes: int) -> np.ndarray:
    """Return predicted probabilities with one column per class index.

    scikit-learn only creates columns for classes seen in training; this puts
    them back at their class index (unseen classes get probability 0), so the
    output always has shape (N, num_classes) as ``src.evaluate.evaluate`` expects.
    """
    proba = model.predict_proba(features)
    out = np.zeros((proba.shape[0], num_classes), dtype=np.float64)
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out


def build_model_a_scratch(cfg: dict, num_classes: int):
    """Model A: custom CNN from scratch, returned as a PrecisionModel."""
    raise NotImplementedError("Model Architect: implement in Week 11")


def build_model_b_frozen(cfg: dict, num_classes: int):
    """Model B: frozen ResNet-50 base + new head, returned as a PrecisionModel."""
    raise NotImplementedError("Model Architect: implement in Week 11")


def build_model_c_finetune(cfg: dict, num_classes: int):
    """Model C: ResNet-50 with last block unfrozen, returned as a PrecisionModel."""
    raise NotImplementedError("Model Architect: implement in Week 11")
