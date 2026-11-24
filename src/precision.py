"""Simulated bf16 weight storage for Keras models.

Research setting studied in this project
----------------------------------------
"bf16 weight storage WITHOUT fp32 master weights":

* forward/backward passes and the optimizer state (Adam moments, momentum)
  stay in float32;
* after every optimizer step, every trainable weight is rounded to the
  nearest value representable in bfloat16 (8 significant bits);
* so any update smaller than half a bf16 ULP of the weight is lost
  ("silent freeze").

Doing the rounding ourselves (instead of using a bf16 dtype policy) means the
experiment runs on any hardware, including Colab T4 GPUs that lack native
bf16 support, and is bit-for-bit deterministic. Keras' built-in
``mixed_bfloat16`` policy keeps fp32 master weights and therefore does NOT
show the effect; it is our "mixed" control condition.

Precision modes
---------------
``fp32``      no rounding (reference).
``bf16_rtn``  round-to-nearest-even after each step (pure bf16 storage).
``bf16_sr``   stochastic rounding after each step (unbiased in expectation).

Weight-decay placement (only used when ``weight_decay > 0``)
-----------------------------------------------------------
``fused``     decoupled decay is added to the optimizer step, then rounded
              once.
``separate``  the optimizer step is rounded, then decay is applied and rounded
              again. This mirrors optimizers that apply decay as its own
              in-place assignment (e.g. Keras AdamW).
"""

from __future__ import annotations

import keras
import tensorflow as tf

PRECISION_MODES = ("fp32", "bf16_rtn", "bf16_sr")
DECAY_MODES = ("fused", "separate")

_LOW_16_BITS_MASK = tf.constant(-65536, dtype=tf.int32)  # 0xFFFF0000


def round_to_bf16(x: tf.Tensor) -> tf.Tensor:
    """Round a float32 tensor to the nearest bfloat16 value (ties to even).

    Args:
        x: float32 tensor.

    Returns:
        float32 tensor whose values are exactly representable in bfloat16.
    """
    return tf.cast(tf.cast(x, tf.bfloat16), tf.float32)


def stochastic_round_to_bf16(x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Stochastically round a float32 tensor to bfloat16.

    bfloat16 is the upper 16 bits of a float32. Adding uniform random noise in
    [0, 2^16) to the raw bits and then zeroing the lower 16 bits rounds the
    magnitude up with probability equal to the discarded fraction, so
    E[round(x)] = x (unbiased).

    Args:
        x: float32 tensor.
        seed: int32 tensor of shape [2] for ``tf.random.stateless_uniform``.

    Returns:
        float32 tensor whose values are exactly representable in bfloat16.
    """
    bits = tf.bitcast(x, tf.int32)
    noise = tf.random.stateless_uniform(
        tf.shape(x), seed=seed, minval=0, maxval=65536, dtype=tf.int32
    )
    rounded = tf.bitwise.bitwise_and(bits + noise, _LOW_16_BITS_MASK)
    return tf.bitcast(rounded, tf.float32)


class PrecisionModel(keras.Model):
    """Functional Keras model whose train step simulates bf16 weight storage.

    Build it like a functional model::

        model = PrecisionModel(inputs, outputs, precision="bf16_rtn")
        model.compile(optimizer="adam", loss=..., metrics=[...])
        model.quantize_weights()  # start from bf16-representable weights
        model.fit(...)

    Extra metrics reported during training:

    * ``freeze_rate``: of the weights the optimizer tried to change this step,
      the fraction that ended up unchanged after rounding (0 for fp32).
    * ``update_ratio``: mean |applied update| / mean |intended update|
      (1.0 means nothing was lost; near 0 means training has stalled).

    Note: only trainable weights are rounded. Non-trainable state such as
    BatchNorm moving statistics is left in float32; freeze BatchNorm with
    ``training=False`` during fine-tuning so those statistics do not change.
    """

    def __init__(
        self,
        *args,
        precision: str = "fp32",
        weight_decay: float = 0.0,
        decay_mode: str = "fused",
        rounding_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if precision not in PRECISION_MODES:
            raise ValueError(f"precision must be one of {PRECISION_MODES}")
        if decay_mode not in DECAY_MODES:
            raise ValueError(f"decay_mode must be one of {DECAY_MODES}")
        self.precision = precision
        self.weight_decay = float(weight_decay)
        self.decay_mode = decay_mode
        self._rounding_seed = rounding_seed
        self._step = tf.Variable(0, dtype=tf.int32, trainable=False, name="rounding_step")
        self.freeze_tracker = keras.metrics.Mean(name="freeze_rate")
        self.update_ratio_tracker = keras.metrics.Mean(name="update_ratio")

    def compile(self, *args, **kwargs):
        """Compile, refusing optimizers that apply their own weight decay.

        Built-in decay (e.g. keras AdamW) runs inside apply_gradients in fp32,
        before our rounding, which would silently turn every run into the
        "fused" case and break the fused-vs-separate ablation. Pass decay via
        PrecisionModel(weight_decay=...) instead.
        """
        super().compile(*args, **kwargs)
        if getattr(self.optimizer, "weight_decay", None):
            raise ValueError(
                "Set the optimizer's weight_decay to None and pass weight_decay= "
                "to PrecisionModel, so decay placement is controlled explicitly."
            )

    # ------------------------------------------------------------------ utils

    def _round(self, x: tf.Tensor, salt: int) -> tf.Tensor:
        """Apply this model's rounding mode to one tensor."""
        if self.precision == "fp32":
            return x
        if self.precision == "bf16_rtn":
            return round_to_bf16(x)
        seed = tf.stack([tf.constant(self._rounding_seed, tf.int32) + salt, self._step])
        return stochastic_round_to_bf16(x, seed)

    def quantize_weights(self) -> None:
        """Round all trainable weights in place (call once before training)."""
        for i, var in enumerate(self.trainable_variables):
            var.assign(self._round(var, salt=i))

    # ------------------------------------------------------------- train step

    def train_step(self, data):
        """One training step: fp32 forward/backward, rounded weight storage."""
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        variables = self.trainable_variables
        old_values = [tf.identity(v) for v in variables]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        lr = tf.cast(self.optimizer.learning_rate, tf.float32)
        wanted = tf.constant(0.0)
        frozen = tf.constant(0.0)
        intended_mag = tf.constant(0.0)
        applied_mag = tf.constant(0.0)

        for i, (var, old) in enumerate(zip(variables, old_values)):
            decay = lr * self.weight_decay * old
            if self.decay_mode == "fused" or self.weight_decay == 0.0:
                intended = var - decay
                new = self._round(intended, salt=2 * i)
            else:  # separate: round optimizer step, then round decay step
                intended = var - decay
                new = self._round(self._round(var, salt=2 * i) - decay, salt=2 * i + 1)
            var.assign(new)

            tried = tf.not_equal(intended, old)
            stuck = tf.logical_and(tried, tf.equal(new, old))
            wanted += tf.reduce_sum(tf.cast(tried, tf.float32))
            frozen += tf.reduce_sum(tf.cast(stuck, tf.float32))
            intended_mag += tf.reduce_sum(tf.abs(intended - old))
            applied_mag += tf.reduce_sum(tf.abs(new - old))

        self._step.assign_add(1)
        self.freeze_tracker.update_state(tf.math.divide_no_nan(frozen, wanted))
        self.update_ratio_tracker.update_state(tf.math.divide_no_nan(applied_mag, intended_mag))

        own = (self.freeze_tracker, self.update_ratio_tracker)
        for metric in self.metrics:  # Keras auto-tracks our two trackers too
            if any(metric is m for m in own):
                continue
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}
