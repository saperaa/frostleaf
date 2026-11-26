"""Phase 1 sanity check: does pure-bf16 weight storage freeze small-LR training?

Trains the same small MLP on synthetic data under every combination of
precision mode x learning rate x optimizer (SGD, SGD+momentum, Adam) and prints:

* loss:       final training loss;
* freeze:     mean share of attempted weight changes that rounded to nothing;
* upd_ratio:  mean over steps of  sum_i |w_i(t+1) - w_i(t)|  /  sum_i |u_i(t)|,
              where u(t) is the intended (fp32) update; 1.0 = nothing lost;
* travelled:  ||w_T - w_0|| of this run divided by the fp32 run's (1.0 for fp32);
* gap_fp32:   ||w_T - w_T(fp32)|| divided by the fp32 run's travel (0.0 for fp32).

Expected (hypothesis H1): with a fine-tuning-sized learning rate (1e-5),
bf16_rtn freezes most updates and barely moves the weights, while bf16_sr
keeps moving them like fp32. With a from-scratch learning rate (1e-3), all
modes train, though round-to-nearest still loses part of each update.

Usage:
    python -m experiments.sanity_bf16_freeze
"""

from __future__ import annotations

import csv
import os

import keras
import numpy as np
from sklearn.datasets import make_classification

from src.precision import PrecisionModel, round_to_bf16

SEED = 42
OUT_CSV = os.path.join(os.path.dirname(__file__), "sanity_bf16_freeze.csv")


def build(precision: str) -> PrecisionModel:
    """Build a fixed-seed 2-hidden-layer MLP (5,699 parameters)."""
    keras.utils.set_random_seed(SEED)
    inputs = keras.Input(shape=(20,))
    h = keras.layers.Dense(64, activation="relu")(inputs)
    h = keras.layers.Dense(64, activation="relu")(h)
    outputs = keras.layers.Dense(3, activation="softmax")(h)
    # The rounding seed follows the run seed, so a different seed also changes
    # the stochastic-rounding noise (otherwise every seed shares one pattern).
    return PrecisionModel(inputs, outputs, precision=precision, rounding_seed=SEED)


def make_optimizer(name: str, lr: float) -> keras.optimizers.Optimizer:
    """Return a fresh optimizer by name."""
    if name == "adam":
        return keras.optimizers.Adam(lr)
    if name == "sgd":
        return keras.optimizers.SGD(lr)
    if name == "sgd_momentum":
        return keras.optimizers.SGD(lr, momentum=0.9)
    raise ValueError(name)


def run(precision: str, lr: float, opt: str, x, y, init_weights):
    """Train one configuration and return summary numbers."""
    model = build(precision)
    model.set_weights(init_weights)
    model.compile(optimizer=make_optimizer(opt, lr), loss="sparse_categorical_crossentropy")
    model.quantize_weights()
    start = [w.copy() for w in model.get_weights()]
    hist = model.fit(x, y, batch_size=128, epochs=5, shuffle=False, verbose=0)
    end = model.get_weights()
    moved = float(np.sqrt(sum(((e - s) ** 2).sum() for s, e in zip(start, end))))
    return end, {
        "loss": hist.history["loss"][-1],
        "freeze_rate": float(np.mean(hist.history["freeze_rate"])),
        "update_ratio": float(np.mean(hist.history["update_ratio"])),
        "distance": moved,
    }


def main() -> None:
    """Run the grid and print/save a results table."""
    x, y = make_classification(
        n_samples=4000, n_features=20, n_informative=10, n_classes=3, random_state=SEED
    )
    x = x.astype("float32")
    # Round the shared start to bf16 once, so the fp32 run starts from exactly the
    # same weights as the bf16 runs and gap_fp32 measures training, not the start.
    init_weights = [round_to_bf16(w).numpy() for w in build("fp32").get_weights()]

    rows = []
    for opt in ("sgd", "sgd_momentum", "adam"):
        for lr in (1e-3, 1e-5):
            ref_dist, ref_end = None, None
            for precision in ("fp32", "bf16_rtn", "bf16_sr"):
                end, r = run(precision, lr, opt, x, y, init_weights)
                if precision == "fp32":
                    ref_dist, ref_end = r["distance"], end
                # distance travelled from the start, relative to the fp32 run
                r["travelled_vs_fp32"] = r["distance"] / ref_dist
                # gap between this run's final weights and the fp32 run's, relative to fp32's travel
                gap = np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(end, ref_end)))
                r["gap_to_fp32_end"] = float(gap) / ref_dist
                rows.append({"optimizer": opt, "lr": lr, "precision": precision, **r})

    header = (
        f"{'optimizer':<13}{'lr':>8}  {'precision':<9}{'loss':>8}{'freeze':>9}"
        f"{'upd_ratio':>11}{'travelled':>11}{'gap_fp32':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['optimizer']:<13}{r['lr']:>8.0e}  {r['precision']:<9}{r['loss']:>8.4f}"
            f"{r['freeze_rate']:>9.1%}{r['update_ratio']:>11.3f}"
            f"{r['travelled_vs_fp32']:>11.3f}{r['gap_to_fp32_end']:>10.3f}"
        )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
