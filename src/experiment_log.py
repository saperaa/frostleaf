"""Experiment log helpers: run IDs, safe appends to experiments/log.csv, run summaries, curves.

Owner: Experimentation Engineer.

Every training run should end with:

    run_id = make_run_id(model_name, change, seed)
    row = {"run_id": run_id, "date": ..., "author": ..., "config_file": ..., ...}
    row.update(history_to_row(history.history))
    append_run(row)
    save_curves(history.history, run_id)

Rules the log follows (and the tests check):
* rows are only appended, never edited or deleted;
* a row is written in the header's column order; missing columns stay empty;
* an unknown column, a missing run_id or a duplicate run_id is an error, not a silent drop;
* a file that has lost its final newline is repaired before appending, so a new
  row never joins the previous line.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_CSV = os.path.join(PROJECT_ROOT, "experiments", "log.csv")
CURVES_DIR = os.path.join(PROJECT_ROOT, "experiments", "curves")


def make_run_id(model: str, change: str, seed: int, when: datetime | None = None) -> str:
    """Return a unique, time-sortable run ID: ``YYYYMMDD-HHMM-<model>-<change>-s<seed>``.

    Args:
        model: short model name, e.g. ``"modelA"``.
        change: the one factor this run changes, e.g. ``"lr1e-3"`` or ``"bf16_sr"``.
        seed: the run's random seed.
        when: timestamp to use; defaults to now (pass it to make the ID reproducible).

    Returns:
        For example ``"20261126-1530-modelA-lr1e-3-s42"``.
    """
    when = when or datetime.now()
    return f"{when.strftime('%Y%m%d-%H%M')}-{model}-{change}-s{seed}"


def _read_header_and_ids(path: str) -> tuple[list[str], set[str]]:
    """Return the log's header and the run IDs already in it."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if "run_id" not in header:
            raise ValueError(f"{path} has no run_id column")
        col = header.index("run_id")
        ids = {row[col] for row in reader if len(row) > col and row[col]}
    return header, ids


def _ends_with_newline(path: str) -> bool:
    """True if the file is empty or its last byte is a newline."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return True
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def append_run(row: dict, path: str = LOG_CSV) -> None:
    """Append one run to the experiment log.

    Args:
        row: column name -> value. Columns not given stay empty.
        path: the log file; its first line is the header.

    Raises:
        ValueError: if ``row`` has a column the header doesn't have, has no
            ``run_id``, or uses a ``run_id`` that is already in the log.
    """
    header, existing_ids = _read_header_and_ids(path)
    unknown = sorted(set(row) - set(header))
    if unknown:
        raise ValueError(f"columns not in the log header: {unknown}")
    run_id = str(row.get("run_id") or "")
    if not run_id:
        raise ValueError("row has no run_id")
    if run_id in existing_ids:
        raise ValueError(f"run_id already in the log: {run_id}")

    needs_newline = not _ends_with_newline(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\r\n")  # csv's own line ending, so the file stays consistent
        csv.DictWriter(f, fieldnames=header).writerow(row)


def history_to_row(history: dict) -> dict:
    """Summarise a Keras ``History.history`` dict into experiment-log columns.

    Args:
        history: per-epoch lists, e.g. ``{"loss": [...], "val_loss": [...],
            "freeze_rate": [...], "update_ratio": [...]}``.

    Returns:
        ``epochs_run``, ``train_loss`` and, when present, ``val_loss`` (last
        epoch), ``mean_freeze_rate`` and ``mean_update_ratio`` (mean over epochs),
        and a ``notes`` entry naming the epoch with the lowest validation loss.
    """
    epochs = len(history["loss"])
    row = {"epochs_run": epochs, "train_loss": float(history["loss"][-1])}
    if "val_loss" in history:
        val = list(history["val_loss"])
        row["val_loss"] = float(val[-1])
        row["notes"] = f"best val_loss epoch {int(np.argmin(val)) + 1}"
    if "freeze_rate" in history:
        row["mean_freeze_rate"] = float(np.mean(history["freeze_rate"]))
    if "update_ratio" in history:
        row["mean_update_ratio"] = float(np.mean(history["update_ratio"]))
    return row


def save_curves(history: dict, run_id: str, out_dir: str = CURVES_DIR) -> str:
    """Save the run's learning curves to ``<out_dir>/<run_id>.png``.

    Panel 1: training and validation loss per epoch. Panel 2 (only when the
    run reports it): the update ratio per epoch, which shows a frozen bf16 run
    that a flat loss curve alone would hide.

    Returns:
        The path written.
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    has_ratio = "update_ratio" in history
    fig = Figure(figsize=(10 if has_ratio else 5.5, 4))
    ax = fig.add_subplot(1, 2 if has_ratio else 1, 1)
    ax.plot(epochs, history["loss"], marker="o", label="train loss")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], marker="o", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    if has_ratio:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(epochs, history["update_ratio"], marker="o", color="tab:red")
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("applied / intended update")
        ax2.set_title("Update ratio (1.0 = nothing lost)")
    for axis in fig.axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
    fig.suptitle(run_id)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.png")
    fig.savefig(path, dpi=120)
    return path
