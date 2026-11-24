"""Unit tests for src/precision.py (run with: pytest -q)."""

import numpy as np
import tensorflow as tf

from src.precision import round_to_bf16, stochastic_round_to_bf16


def _is_bf16_representable(x: np.ndarray) -> bool:
    bits = x.astype(np.float32).view(np.uint32)
    return bool(np.all((bits & 0xFFFF) == 0))


def test_rtn_outputs_are_bf16_values():
    x = tf.random.normal([1000], seed=1)
    assert _is_bf16_representable(round_to_bf16(x).numpy())


def test_rtn_error_is_within_half_ulp():
    x = tf.random.uniform([1000], 0.03125, 0.0625, seed=2)  # one binade: ULP = 2**-12
    err = tf.abs(round_to_bf16(x) - x).numpy()
    assert err.max() <= 2.0**-13 + 1e-12


def test_rtn_swallows_tiny_update():
    w = round_to_bf16(tf.constant([0.05]))
    updated = round_to_bf16(w - 1e-5)  # 1e-5 < half ULP (~1.2e-4) at 0.05
    assert updated.numpy()[0] == w.numpy()[0]


def test_sr_outputs_are_bf16_values():
    x = tf.random.normal([1000], seed=3)
    out = stochastic_round_to_bf16(x, tf.constant([0, 0])).numpy()
    assert _is_bf16_representable(out)


def test_sr_is_unbiased_on_tiny_update():
    w = float(round_to_bf16(tf.constant([0.05])).numpy()[0])
    target = tf.fill([200_000], w - 1e-5)
    out = stochastic_round_to_bf16(target, tf.constant([7, 0])).numpy()
    assert abs(out.mean() - (w - 1e-5)) < 2e-6  # mean preserved, unlike RTN


def test_sr_handles_negative_values():
    x = tf.constant([-0.05 - 1e-5] * 100_000)
    out = stochastic_round_to_bf16(x, tf.constant([3, 1])).numpy()
    assert np.all(out < 0)
    assert abs(out.mean() - (-0.05 - 1e-5)) < 2e-6


def test_builtin_weight_decay_is_rejected():
    import keras
    import pytest

    from src.precision import PrecisionModel

    inputs = keras.Input(shape=(4,))
    model = PrecisionModel(inputs, keras.layers.Dense(2)(inputs), precision="bf16_rtn")
    with pytest.raises(ValueError):
        model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-2), loss="mse")


def test_separate_decay_is_swallowed_but_fused_is_not():
    """At lr=1e-3, wd=1e-2 the decay step (~5e-7) is far below half a bf16 ULP.

    Separate rounding must erase it; fused rounding with a larger optimizer
    step must keep part of it (dithering), so the two weights must differ.
    """
    import keras
    import numpy as np

    from src.precision import PrecisionModel

    x = np.random.default_rng(0).normal(size=(512, 8)).astype("float32")
    y = x @ np.ones((8, 1), "float32")
    finals = {}
    for mode in ("fused", "separate"):
        keras.utils.set_random_seed(0)
        inputs = keras.Input(shape=(8,))
        model = PrecisionModel(
            inputs, keras.layers.Dense(1)(inputs),
            precision="bf16_rtn", weight_decay=1e-2, decay_mode=mode,
        )
        model.compile(optimizer=keras.optimizers.SGD(1e-3), loss="mse")
        model.quantize_weights()
        model.fit(x, y, batch_size=64, epochs=3, shuffle=False, verbose=0)
        finals[mode] = model.get_weights()[0]
    assert not np.array_equal(finals["fused"], finals["separate"])
