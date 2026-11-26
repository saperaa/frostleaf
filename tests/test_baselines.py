"""Unit tests for the two baselines (src/models.py, src/baselines.py)."""

import json
import os

import numpy as np
import tensorflow as tf

from src.baselines import dataset_to_arrays, run_baselines
from src.models import full_width_proba, logistic_regression_baseline, majority_class_baseline


def test_majority_baseline_predicts_most_common_class():
    y = np.array([2, 2, 2, 0, 1])
    model = majority_class_baseline(y)
    proba = full_width_proba(model, np.zeros((4, 7)), num_classes=4)
    assert proba.shape == (4, 4)
    assert (proba.argmax(axis=1) == 2).all()


def test_full_width_proba_places_unseen_classes_at_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5))
    y = np.repeat([0, 2, 3], 20)  # class 1 never seen in training
    model = logistic_regression_baseline(x, y)
    proba = full_width_proba(model, x, num_classes=4)
    assert proba.shape == (60, 4)
    assert np.allclose(proba[:, 1], 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_scaler_is_fitted_on_train_only():
    rng = np.random.default_rng(1)
    train_x = rng.normal(0, 1, size=(100, 3))
    model = logistic_regression_baseline(train_x, rng.integers(0, 2, 100))
    assert np.allclose(model[0].mean_, train_x.mean(axis=0))


def test_dataset_to_arrays_handles_batched_uint8_images():
    images = np.random.default_rng(2).integers(0, 256, size=(10, 64, 64, 3), dtype=np.uint8)
    labels = np.arange(10) % 3
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
    x, y = dataset_to_arrays(ds, size=8)
    assert x.shape == (10, 8 * 8 * 3)
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert (y == labels).all()


def test_run_baselines_learned_beats_majority(tmp_path):
    """On separable data the learned baseline must beat the trivial one."""
    rng = np.random.default_rng(3)
    centres = rng.normal(0, 3, size=(4, 12))
    y = rng.integers(0, 4, 400)
    x = centres[y] + rng.normal(0, 0.5, size=(400, 12))
    out = run_baselines(x[:300], y[:300], x[300:], y[300:], list("abcd"), "test",
                        out_dir=str(tmp_path), log_csv=None)
    assert out["logistic_regression"]["val_macro_f1"] > out["majority_class"]["val_macro_f1"]
    assert os.path.exists(tmp_path / "baseline_results.json")
    assert os.path.exists(tmp_path / "logistic_regression_val_cm.png")
    json.loads((tmp_path / "baseline_results.json").read_text())


def test_run_baselines_logs_two_rows_through_append_run(tmp_path):
    """Both baselines are logged with the real log header, one row each."""
    import csv

    from src.experiment_log import LOG_CSV

    log = tmp_path / "log.csv"
    log.write_text(open(LOG_CSV).readline())  # header only
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(120, 6)), rng.integers(0, 3, 120)
    run_baselines(x[:90], y[:90], x[90:], y[90:], list("abc"), "test",
                  out_dir=str(tmp_path / "out"), log_csv=str(log))
    rows = list(csv.DictReader(open(log)))
    assert [r["model"] for r in rows] == ["baseline_majority_class", "baseline_logistic_regression"]
    assert len({r["run_id"] for r in rows}) == 2


def test_main_loads_plantvillage_without_augmentation(monkeypatch):
    """Baselines must train on plain images, never on augmented ones."""
    import src.baselines as baselines
    import src.data

    seen = {}

    def fake_load(variant, cfg=None):
        seen["augmentation"] = cfg["data"]["augmentation"]
        raise SystemExit  # stop before any real data is needed

    monkeypatch.setattr(src.data, "load_plantvillage", fake_load)
    monkeypatch.setattr("sys.argv", ["baselines", "--author", "test"])
    try:
        baselines.main()
    except SystemExit:
        pass
    assert seen["augmentation"] == []
