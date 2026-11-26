"""Unit tests for src/evaluate.py on hand-made predictions (run with: pytest -q)."""

import math

import numpy as np
import pytest

from src.evaluate import evaluate, normalize_rows, plot_confusion_matrix

NAMES = ["a", "b", "c"]


def _one_hot(labels, n_classes=3):
    """Return probabilities that put all mass on the given class per row."""
    prob = np.zeros((len(labels), n_classes))
    prob[np.arange(len(labels)), labels] = 1.0
    return prob


def test_perfect_predictor_scores_one():
    """A predictor that is always right gets accuracy, macro-F1 and AUC of 1."""
    y = np.array([0, 1, 2, 0, 1, 2])
    m = evaluate(y, _one_hot(y), NAMES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["roc_auc_macro"] == 1.0
    assert np.array_equal(m["confusion_matrix"], np.diag([2, 2, 2]))
    assert m["auc_skipped_classes"] == [] and m["absent_classes"] == []


def test_constant_predictor_matches_hand_computation():
    """Always predicting class 0 on y = [0, 0, 1, 2].

    Class a: precision 2/4, recall 2/2, F1 = 2*0.5*1/1.5 = 2/3.
    Classes b, c: never predicted, F1 = 0. Macro-F1 = (2/3)/3 = 2/9.
    Accuracy = 2/4.
    """
    y = np.array([0, 0, 1, 2])
    m = evaluate(y, _one_hot([0, 0, 0, 0]), NAMES)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(2 / 9)
    assert m["per_class"]["a"]["precision"] == pytest.approx(0.5)
    assert m["per_class"]["a"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["b"]["f1"] == 0.0
    assert m["per_class"]["c"]["support"] == 1


def test_missing_class_does_not_crash_auc():
    """Class c never occurs: its AUC is skipped and reported, not an error."""
    y = np.array([0, 1, 0, 1])
    prob = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4]])
    m = evaluate(y, prob, NAMES)
    assert m["auc_skipped_classes"] == ["c"]
    assert m["absent_classes"] == ["c"]
    assert set(m["roc_auc_per_class"]) == {"a", "b"}
    assert math.isnan(m["per_class"]["c"]["roc_auc"])
    assert not math.isnan(m["roc_auc_macro"])


def test_macro_f1_ignores_absent_classes():
    """An absent class is not counted as F1 = 0 in the macro average."""
    y = np.array([0, 1, 0, 1])
    m = evaluate(y, _one_hot(y), NAMES)
    assert m["macro_f1"] == 1.0
    assert math.isnan(m["per_class"]["c"]["f1"])


def test_bad_shapes_raise():
    """Mismatched probability shape or out-of-range labels are rejected."""
    with pytest.raises(ValueError):
        evaluate([0, 1], np.zeros((3, 3)), NAMES)
    with pytest.raises(ValueError):
        evaluate([0, 5], np.zeros((2, 3)), NAMES)


def test_normalize_rows_gives_recall_on_diagonal():
    """Each non-empty row sums to 1; an empty row becomes NaN."""
    cm = np.array([[3, 1, 0], [0, 2, 2], [0, 0, 0]])
    norm = normalize_rows(cm)
    assert norm[0, 0] == pytest.approx(0.75)
    assert np.allclose(norm[:2].sum(axis=1), 1.0)
    assert np.all(np.isnan(norm[2]))


def test_plot_confusion_matrix_writes_file(tmp_path):
    """A 38-class heatmap with an empty row is saved without error."""
    rng = np.random.default_rng(0)
    cm = rng.integers(0, 20, size=(38, 38))
    cm[5] = 0
    names = [f"class_{i}" for i in range(38)]
    out = plot_confusion_matrix(cm, names, str(tmp_path / "sub" / "cm.png"))
    assert (tmp_path / "sub" / "cm.png").stat().st_size > 0
    assert out.endswith("cm.png")
