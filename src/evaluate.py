"""Test-set evaluation and error analysis.

Owner: Evaluation & Deployment Engineer.

On the PlantVillage test split AND the PlantDoc shared-class set, for the
final model and both baselines: accuracy, confusion matrix, per-class
precision/recall/F1, macro-F1 (primary metric), one-vs-rest ROC + AUC.

Error analysis: inspect >= 20 misclassified samples, group them into failure
categories, show predicted probabilities and Grad-CAM heatmaps, and propose
one concrete fix for the most common confusion.
"""

from __future__ import annotations


def evaluate(model, dataset, class_names: list[str]) -> dict:
    """Compute the full metric set for one model on one dataset."""
    raise NotImplementedError("Evaluation Engineer: implement in Week 13")


def worst_predictions(model, dataset, k: int = 20):
    """Return the k most confidently wrong samples for error analysis."""
    raise NotImplementedError("Evaluation Engineer: implement in Week 13")
