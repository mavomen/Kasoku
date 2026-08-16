"""LSTM forecaster architecture.

The model is built with the Keras Functional API so the same object runs in
eager mode, under ``tf.function``, and under ``tf.function(jit_compile=True)``
(XLA) without any mode-specific code paths. Passing ``seed`` seeds every
random initializer so identical builds produce identical weights.
"""

from __future__ import annotations

import tensorflow as tf

from kasoku.models.base import ModelError

LSTM_SIZES: dict[str, dict[str, int | float]] = {
    "small": {"hidden_units": 32, "num_layers": 1, "dropout": 0.1},
    "medium": {"hidden_units": 64, "num_layers": 2, "dropout": 0.2},
}


def build_lstm(
    lookback: int,
    horizon: int,
    *,
    hidden_units: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    name: str = "lstm_forecaster",
    seed: int | None = None,
) -> tf.keras.Model:
    """Build an LSTM forecaster: stacked LSTM layers over a Dense head.

    Input shape is ``(None, lookback, 1)`` and output shape ``(None, horizon)``.
    When ``seed`` is given, all random initializers are seeded so that
    repeated builds with the same seed yield identical weights.
    """
    if lookback < 1:
        raise ModelError(f"lookback must be >= 1, got {lookback}")
    if horizon < 1:
        raise ModelError(f"horizon must be >= 1, got {horizon}")
    if num_layers < 1:
        raise ModelError(f"num_layers must be >= 1, got {num_layers}")
    if hidden_units < 1:
        raise ModelError(f"hidden_units must be >= 1, got {hidden_units}")

    inputs = tf.keras.Input(shape=(lookback, 1), name="window")
    x = inputs
    for index in range(num_layers):
        x = tf.keras.layers.LSTM(
            hidden_units,
            return_sequences=index < num_layers - 1,
            dropout=dropout,
            name=f"lstm_{index}",
            **_initializers(seed),
        )(x)
    outputs = tf.keras.layers.Dense(
        horizon,
        name="output",
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(seed=seed) if seed is not None else None
        ),
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_lstm_by_size(
    size: str,
    lookback: int,
    horizon: int,
    *,
    seed: int | None = None,
) -> tf.keras.Model:
    """Build an LSTM forecaster from a named size config (see :data:`LSTM_SIZES`)."""
    try:
        config = LSTM_SIZES[size]
    except KeyError as exc:
        raise ModelError(
            f"unknown lstm size {size!r}; expected one of {sorted(LSTM_SIZES)}"
        ) from exc
    return build_lstm(
        lookback,
        horizon,
        hidden_units=int(config["hidden_units"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        name=f"lstm_{size}",
        seed=seed,
    )


def _initializers(seed: int | None) -> dict:
    if seed is None:
        return {}
    return {
        "kernel_initializer": tf.keras.initializers.GlorotUniform(seed=seed),
        "recurrent_initializer": tf.keras.initializers.Orthogonal(seed=seed),
    }
