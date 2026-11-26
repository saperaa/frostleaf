"""Unit tests for src/experiment_log.py (run with: pytest -q)."""

import csv
from datetime import datetime

import pytest

from src.experiment_log import append_run, history_to_row, make_run_id, save_curves


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_run_id_format():
    run_id = make_run_id("modelA", "lr1e-3", 42, datetime(2026, 11, 26, 15, 30))
    assert run_id == "20261126-1530-modelA-lr1e-3-s42"


def test_row_written_in_header_order_with_missing_columns_empty(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("run_id,a,b,c\n")
    append_run({"c": 3, "run_id": "r1", "a": 1}, str(log))
    assert _rows(log) == [["run_id", "a", "b", "c"], ["r1", "1", "", "3"]]


def test_unknown_column_is_rejected(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("run_id,val_macro_f1\n")
    with pytest.raises(ValueError, match="val_macro_F1"):
        append_run({"run_id": "r1", "val_macro_F1": 0.5}, str(log))
    assert _rows(log) == [["run_id", "val_macro_f1"]]  # nothing was written


def test_missing_final_newline_is_repaired(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("run_id,val_macro_f1")  # no newline at the end
    append_run({"run_id": "r1", "val_macro_f1": 0.5}, str(log))
    assert _rows(log) == [["run_id", "val_macro_f1"], ["r1", "0.5"]]


def test_missing_or_duplicate_run_id_is_rejected(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("run_id,val_macro_f1\n")
    with pytest.raises(ValueError, match="no run_id"):
        append_run({"val_macro_f1": 0.5}, str(log))
    append_run({"run_id": "r1"}, str(log))
    with pytest.raises(ValueError, match="already in the log"):
        append_run({"run_id": "r1"}, str(log))
    assert len(_rows(log)) == 2


def test_history_to_row_matches_hand_computation():
    history = {
        "loss": [1.20, 0.90, 0.80],
        "val_loss": [1.10, 1.00, 1.05],
        "freeze_rate": [0.97, 0.96, 0.95],
        "update_ratio": [0.03, 0.04, 0.05],
    }
    row = history_to_row(history)
    assert row["epochs_run"] == 3
    assert row["train_loss"] == pytest.approx(0.80)
    assert row["val_loss"] == pytest.approx(1.05)
    assert row["mean_freeze_rate"] == pytest.approx(0.96)
    assert row["mean_update_ratio"] == pytest.approx(0.04)
    assert row["notes"] == "best val_loss epoch 2"


def test_history_to_row_without_precision_metrics():
    row = history_to_row({"loss": [0.5, 0.4]})
    assert row == {"epochs_run": 2, "train_loss": pytest.approx(0.4)}


def test_save_curves_writes_png(tmp_path):
    history = {"loss": [1.2, 0.9, 0.8], "val_loss": [1.1, 1.0, 1.05], "update_ratio": [0.03, 0.04, 0.05]}
    path = save_curves(history, "20261126-1530-modelA-lr1e-3-s42", str(tmp_path / "curves"))
    assert path.endswith("20261126-1530-modelA-lr1e-3-s42.png")
    assert (tmp_path / "curves" / "20261126-1530-modelA-lr1e-3-s42.png").stat().st_size > 0


def test_works_with_the_real_log_header(tmp_path):
    real_header = open("experiments/log.csv", encoding="utf-8").readline()
    log = tmp_path / "log.csv"
    log.write_text(real_header)
    row = {"run_id": make_run_id("modelA", "lr1e-3", 42, datetime(2026, 11, 26, 15, 30)), "seed": 42}
    row.update(history_to_row({"loss": [1.2, 0.8], "val_loss": [1.1, 1.05]}))
    append_run(row, str(log))
    header, written = _rows(log)
    assert len(written) == len(header) == 25
    assert written[header.index("epochs_run")] == "2"
