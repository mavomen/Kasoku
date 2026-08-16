"""Temporal convolutional network (TCN) forecaster architecture.

The model is built with the Keras Functional API so the same object runs in
eager mode, under ``tf.function``, and under ``tf.function(jit_compile=True)``
(XLA) without any mode-specific code paths. Passing ``seed`` seeds every
random initializer so identical builds produce identical weights.

The network stacks residual blocks of causal dilated convolutions (Bai et al.,
2018); each block is a shortcut (1x1 projection when channel counts differ)
summed with two causal ``Conv1D`` layers separated by ReLU and dropout.
"""

from __future__ import annotations

import tensorflow as tf

from kasoku.models.base import ModelError

TCN_SIZES: dict[str, dict[str, int | float]] = {
    "small": {
        "filters": 32,
        "num_blocks": 1,
        "kernel_size": 3,
        "dilation_base": 2,
        "dropout": 0.1,
    },
    "medium": {
        "filters": 64,
        "num_blocks": 2,
        "kernel_size": 3,
        "dilation_base": 2,
        "dropout": 0.2,
    },
}


def build_tcn(
    lookback: int,
    horizon: int,
    *,
    filters: int = 32,
    num_blocks: int = 1,
    kernel_size: int = 3,
    dilation_base: int = 2,
    dropout: float = 0.1,
    name: str = "tcn_forecaster",
    seed: int | None = None,
) -> tf.keras.Model:
    """Build a TCN forecaster: stacked residual blocks over a Dense head.

    Input shape is ``(None, lookback, 1)`` and output shape ``(None, horizon)``.
    Block ``i`` uses dilation ``dilation_base ** i``. When ``seed`` is given,
    all random initializers and the global RNG state are seeded so that
    repeated builds with the same seed yield identical weights and
    deterministic training (including dropout RNGs).
    """
    if lookback < 1:
        raise ModelError(f"lookback must be >= 1, got {lookback}")
    if horizon < 1:
        raise ModelError(f"horizon must be >= 1, got {horizon}")
    if filters < 1:
        raise ModelError(f"filters must be >= 1, got {filters}")
    if num_blocks < 1:
        raise ModelError(f"num_blocks must be >= 1, got {num_blocks}")
    if kernel_size < 1:
        raise ModelError(f"kernel_size must be >= 1, got {kernel_size}")
    if dilation_base < 1:
        raise ModelError(f"dilation_base must be >= 1, got {dilation_base}")
    if not 0.0 <= dropout < 1.0:
        raise ModelError(f"dropout must be in [0, 1), got {dropout}")
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.Input(shape=(lookback, 1), name="window")
    x = inputs
    for index in range(num_blocks):
        x = _residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=dilation_base**index,
            dropout=dropout,
            seed=seed,
            name=f"block_{index}",
        )
    last = x[:, -1, :]
    outputs = tf.keras.layers.Dense(
        horizon,
        name="output",
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(seed=seed) if seed is not None else None
        ),
    )(last)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_tcn_by_size(
    size: str,
    lookback: int,
    horizon: int,
    *,
    seed: int | None = None,
) -> tf.keras.Model:
    """Build a TCN forecaster from a named size config (see :data:`TCN_SIZES`)."""
    try:
        config = TCN_SIZES[size]
    except KeyError as exc:
        raise ModelError(
            f"unknown tcn size {size!r}; expected one of {sorted(TCN_SIZES)}"
        ) from exc
    return build_tcn(
        lookback,
        horizon,
        filters=int(config["filters"]),
        num_blocks=int(config["num_blocks"]),
        kernel_size=int(config["kernel_size"]),
        dilation_base=int(config["dilation_base"]),
        dropout=float(config["dropout"]),
        name=f"tcn_{size}",
        seed=seed,
    )


def _residual_block(
    x,
    *,
    filters: int,
    kernel_size: int,
    dilation: int,
    dropout: float,
    seed: int | None,
    name: str,
):
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters,
            1,
            padding="same",
            name=f"{name}_proj",
            kernel_initializer=_kernel_initializer(seed),
        )(x)
    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation,
        name=f"{name}_conv1",
        kernel_initializer=_kernel_initializer(seed),
    )(x)
    y = tf.keras.layers.ReLU(name=f"{name}_relu1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout1")(y)
    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation,
        name=f"{name}_conv2",
        kernel_initializer=_kernel_initializer(seed),
    )(y)
    y = tf.keras.layers.ReLU(name=f"{name}_relu2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout2")(y)
    return tf.keras.layers.Add(name=f"{name}_add")([y, shortcut])


def _kernel_initializer(seed: int | None):
    if seed is None:
        return None
    return tf.keras.initializers.GlorotUniform(seed=seed)
