"""Forecast inference: point predictions, multi-step recursion, and checkpoints.

Forecasting is execution-mode agnostic: the model is wrapped once via
``kasoku.execution.compile_call`` for the requested mode, and checkpoints saved
by :func:`kasoku.train.train` are plain weights, so a model trained under XLA
loads and predicts in eager mode (or any other).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from kasoku.execution import compile_call
from kasoku.forecast.base import ForecastError
from kasoku.models import build_gru_by_size, build_lstm_by_size, build_tcn_by_size
from kasoku.pipeline import validate_canonical
from kasoku.train import load_checkpoint

FORECAST_COLUMNS = ["timestamp", "symbol_or_series_id", "horizon_step", "predicted_value", "model"]

_MODEL_BUILDERS = {
    "gru": build_gru_by_size,
    "lstm": build_lstm_by_size,
    "tcn": build_tcn_by_size,
}


def forecast_point(
    model: tf.keras.Model,
    window: np.ndarray,
    *,
    mode: str = "eager",
) -> np.ndarray:
    """Predict the model's native horizon from a single window.

    ``window`` has shape ``(lookback, 1)``; the returned array has shape
    ``(horizon,)``. The model is wrapped for the requested execution mode.
    """
    lookback = int(model.input_shape[1])
    win = _as_window(window, lookback)
    predict = compile_call(model, mode)
    return predict(win[None, ...]).numpy()[0]


def forecast_multi_step(
    model: tf.keras.Model,
    window: np.ndarray,
    steps: int,
    *,
    mode: str = "eager",
) -> np.ndarray:
    """Recursively forecast ``steps`` values ahead of ``window``.

    The model's native horizon is used as the recursive step size: each pass
    predicts the next horizon values, which are appended to the causal window
    before the next pass. Returns an array of exactly ``steps`` values.
    """
    if steps < 1:
        raise ForecastError(f"steps must be >= 1, got {steps}")
    lookback = int(model.input_shape[1])
    horizon = int(model.output_shape[1])
    win = _as_window(window, lookback)
    predict = compile_call(model, mode)
    out: list[float] = []
    while len(out) < steps:
        prediction = predict(win[None, ...]).numpy()[0]
        out.extend(float(value) for value in prediction)
        win = np.concatenate([win, prediction[:, None]], axis=0)[-lookback:]
    return np.asarray(out[:steps], dtype=np.float32)


def forecast_series(
    model: tf.keras.Model,
    df: pd.DataFrame,
    steps: int,
    *,
    mode: str = "eager",
    freq: str | None = None,
) -> pd.DataFrame:
    """Forecast each series in a canonical ``df`` ``steps`` ahead.

    Returns a frame matching the forecast output schema (timestamp,
    symbol_or_series_id, horizon_step, predicted_value, model). Future
    timestamps continue from each series' last observed timestamp using
    ``freq`` (inferred via ``pd.infer_freq`` with a median-interval fallback
    when not given).
    """
    validate_canonical(df)
    lookback = int(model.input_shape[1])
    rows = []
    for symbol, group in df.groupby("symbol_or_series_id", sort=True):
        group = group.sort_values("timestamp")
        window = group["value"].to_numpy(dtype=np.float32)[-lookback:, None]
        predictions = forecast_multi_step(model, window, steps, mode=mode)
        timestamps = _future_timestamps(group["timestamp"], steps, freq)
        for step, (timestamp, value) in enumerate(zip(timestamps, predictions), start=1):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol_or_series_id": symbol,
                    "horizon_step": step,
                    "predicted_value": float(value),
                    "model": model.name,
                }
            )
    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def _future_timestamps(
    index: pd.Series,
    steps: int,
    freq: str | None,
) -> pd.DatetimeIndex:
    if freq is None:
        freq = _infer_freq(index)
    return pd.date_range(start=index.iloc[-1], periods=steps + 1, freq=freq)[1:]


def _infer_freq(index: pd.Series) -> str | pd.Timedelta:
    if len(index) >= 3:
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return inferred
    median = index.diff().median()
    if pd.isna(median):
        raise ForecastError(
            "could not infer a timestamp frequency for the series; pass freq explicitly"
        )
    return median


def load_trained_model(
    model_name: str,
    size: str,
    lookback: int,
    horizon: int,
    checkpoint_path: str | Path,
    *,
    seed: int | None = None,
) -> tf.keras.Model:
    """Rebuild a model architecture and load checkpoint weights into it.

    Checkpoints are plain weights saved by :func:`kasoku.train.train`, which
    are independent of the execution mode used to train, so a model trained
    under XLA (or ``tf.function``) loads and predicts in any mode here.
    """
    try:
        builder = _MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ForecastError(
            f"unknown model {model_name!r}; expected one of {sorted(_MODEL_BUILDERS)}"
        ) from exc
    model = builder(size, lookback=lookback, horizon=horizon, seed=seed)
    load_checkpoint(model, checkpoint_path)
    return model


def _as_window(window: np.ndarray, lookback: int) -> np.ndarray:
    win = np.asarray(window, dtype=np.float32)
    if win.shape != (lookback, 1):
        raise ForecastError(
            f"window must have shape ({lookback}, 1), got {win.shape}"
        )
    return win
