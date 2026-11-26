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


def _tiny_regression_data(n=64, mean=0.0):
    """Small deterministic regression problem: y = sum(x)."""
    x = np.random.default_rng(0).normal(loc=mean, size=(n, 4)).astype("float32")
    return x, x.sum(axis=1, keepdims=True)


def _fit_two_layer(variable_groups_fn=None, steps=4):
    """Train a fixed-seed backbone/head model; return (model, last-epoch metrics)."""
    import keras

    from src.precision import PrecisionModel

    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(4,))
    # Backbone weights are large (in [4, 8): half a bf16 ULP = 2**-6), head
    # weights are small (~0.015: half ULP ~3e-5), so an Adam step of ~1e-4
    # vanishes in the backbone but survives in the head.
    backbone = keras.layers.Dense(
        4, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(4.0, 7.9)
    )
    head = keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomUniform(0.01, 0.015))
    groups = variable_groups_fn(backbone, head) if variable_groups_fn else None
    model = PrecisionModel(
        inputs, head(backbone(inputs)), precision="bf16_rtn", variable_groups=groups
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")
    model.quantize_weights()
    x, y = _tiny_regression_data()
    history = model.fit(x, y, batch_size=64 // steps, epochs=1, shuffle=False, verbose=0)
    return model, {k: v[-1] for k, v in history.history.items()}


def test_default_metrics_unchanged_without_groups():
    """Without variable_groups the metrics are as before; with them, overall values match."""
    _, plain = _fit_two_layer()
    assert set(plain) == {"loss", "freeze_rate", "update_ratio"}
    _, grouped = _fit_two_layer(lambda b, h: {"backbone": [b], "head": [h]})
    for key in plain:  # grouping only adds metrics; it never changes training
        assert grouped[key] == plain[key]


def test_head_and_backbone_freeze_differently():
    """Large backbone weights freeze at lr 1e-4 while small head weights keep learning."""
    _, metrics = _fit_two_layer(lambda b, h: {"backbone": [b], "head": [h]})
    assert {"freeze_rate_backbone", "update_ratio_backbone",
            "freeze_rate_head", "update_ratio_head"} <= set(metrics)
    assert metrics["freeze_rate_backbone"] == 1.0  # every backbone update lost
    assert metrics["update_ratio_backbone"] == 0.0
    assert metrics["freeze_rate_head"] < 0.5  # most head updates survive
    assert metrics["freeze_rate_head"] < metrics["freeze_rate"] < 1.0


def test_variable_in_two_groups_is_rejected():
    """A variable may belong to at most one group."""
    import pytest

    with pytest.raises(ValueError):
        _fit_two_layer(lambda b, h: {"a": [b, h], "b": [h]})


def test_batchnorm_stats_frozen_with_training_false():
    """BatchNorm called with training=False must keep its moving statistics.

    PrecisionModel only rounds trainable weights; moving_mean/moving_variance
    are updated by the layer itself. Model C relies on training=False to keep
    the ImageNet statistics: if they drifted, the fine-tuned model would be
    evaluated with different normalisation than it was trained with.
    """
    import keras

    from src.precision import PrecisionModel

    def build(bn_training):
        """Tiny Dense -> BatchNorm -> Dense model; return it and its BN layer."""
        keras.utils.set_random_seed(0)
        inputs = keras.Input(shape=(4,))
        bn = keras.layers.BatchNormalization()
        hidden = bn(keras.layers.Dense(8)(inputs), training=bn_training)
        model = PrecisionModel(inputs, keras.layers.Dense(1)(hidden), precision="bf16_rtn")
        model.compile(optimizer="adam", loss="mse")
        model.quantize_weights()
        return model, bn

    x, y = _tiny_regression_data(n=80, mean=5.0)  # mean far from BN's initial 0

    model, bn = build(bn_training=False)
    before = [bn.moving_mean.numpy().copy(), bn.moving_variance.numpy().copy()]
    model.fit(x, y, batch_size=16, epochs=1, shuffle=False, verbose=0)  # 5 steps
    assert model.optimizer.iterations.numpy() == 5
    np.testing.assert_array_equal(bn.moving_mean.numpy(), before[0])
    np.testing.assert_array_equal(bn.moving_variance.numpy(), before[1])

    # Control: without training=False the same 5 steps DO move the statistics,
    # so the assertions above would catch the bug.
    model, bn = build(bn_training=None)
    before_mean = bn.moving_mean.numpy().copy()
    model.fit(x, y, batch_size=16, epochs=1, shuffle=False, verbose=0)
    assert not np.array_equal(bn.moving_mean.numpy(), before_mean)


def test_nested_base_training_false_but_dropout_still_active():
    """Model C pattern: base(x, training=False) then a Dropout head.

    The nested base model's BatchNorm statistics must stay fixed, while the
    head's Dropout (built without an explicit training flag) must still be
    active during fit().
    """
    import keras

    from src.precision import PrecisionModel

    keras.utils.set_random_seed(0)
    base_in = keras.Input(shape=(4,))
    bn = keras.layers.BatchNormalization()
    base = keras.Model(base_in, bn(keras.layers.Dense(8)(base_in)))

    inputs = keras.Input(shape=(4,))
    hidden = keras.layers.Dropout(0.5)(base(inputs, training=False))
    model = PrecisionModel(inputs, keras.layers.Dense(1)(hidden), precision="bf16_rtn")
    model.compile(optimizer="adam", loss="mse")
    model.quantize_weights()

    x, y = _tiny_regression_data(n=80, mean=5.0)
    model.fit(x, y, batch_size=16, epochs=1, shuffle=False, verbose=0)
    np.testing.assert_array_equal(bn.moving_mean.numpy(), np.zeros(8, "float32"))

    in_training = model(x[:8], training=True).numpy()
    in_inference = model(x[:8], training=False).numpy()
    assert not np.allclose(in_training, in_inference)  # dropout is on in training


def test_train_step_compiles_with_xla_and_matches_eager_graph():
    """jit_compile=True (Keras' GPU default on Colab) must work and give the same
    weights as the non-XLA step: stochastic rounding uses stateless seeds."""
    import keras
    import numpy as np

    from src.precision import PrecisionModel

    x = np.random.default_rng(0).normal(size=(256, 8)).astype("float32")
    y = (x[:, 0] > 0).astype("int32")
    finals = {}
    for jit in (False, True):
        keras.utils.set_random_seed(0)
        inputs = keras.Input(shape=(8,))
        model = PrecisionModel(inputs, keras.layers.Dense(2, activation="softmax")(inputs),
                               precision="bf16_sr", rounding_seed=3)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy", jit_compile=jit)
        model.quantize_weights()
        model.fit(x, y, batch_size=32, epochs=2, shuffle=False, verbose=0)
        finals[jit] = model.get_weights()[0]
    assert np.array_equal(finals[False], finals[True])
