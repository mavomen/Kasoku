"""Unit tests for point and multi-step forecast correctness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kasoku.data import CsvSource
from kasoku.forecast import (
    FORECAST_COLUMNS,
    ForecastError,
    forecast_multi_step,
    forecast_point,
    forecast_series,
    load_trained_model,
)
from kasoku.models import build_gru_by_size, build_lstm_by_size
from kasoku.pipeline import PipelineError
from kasoku.train import train

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"
MULTI = FIXTURES / "series_multi.csv"


def _window_df():
    return CsvSource(SINGLE).read()


def _last_window(df=None, lookback=24):
    df = df if df is not None else _window_df()
    return df["value"].to_numpy(dtype=np.float32)[-lookback:, None]


def test_forecast_point_shape_and_finite():
    model = build_gru_by_size("small", 24, 1, seed=42)
    pred = forecast_point(model, _last_window())
    assert pred.shape == (1,)
    assert np.isfinite(pred).all()
    assert pred.dtype == np.float32


def test_forecast_point_equals_direct_call():
    model = build_gru_by_size("small", 24, 1, seed=42)
    win = _last_window()
    expected = model(win[None, ...]).numpy()[0]
    np.testing.assert_allclose(forecast_point(model, win), expected, rtol=1e-6)


def test_forecast_point_bad_window_raises():
    model = build_gru_by_size("small", 24, 1, seed=42)
    with pytest.raises(ForecastError, match="window must have shape"):
        forecast_point(model, np.zeros((10, 1)))
    with pytest.raises(ForecastError, match="window must have shape"):
        forecast_point(model, np.zeros((24,)))


def test_forecast_point_mode_equivalence():
    model = build_gru_by_size("small", 24, 1, seed=42)
    win = _last_window()
    eager = forecast_point(model, win, mode="eager")
    xla = forecast_point(model, win, mode="xla")
    np.testing.assert_allclose(eager, xla, rtol=1e-5, atol=1e-5)


def test_forecast_multi_step_horizon_one():
    model = build_gru_by_size("small", 24, 1, seed=42)
    pred = forecast_multi_step(model, _last_window(), steps=5)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred[:1], forecast_point(model, _last_window()), atol=1e-6)


def test_forecast_multi_step_horizon_three():
    model = build_gru_by_size("medium", 24, 3, seed=42)
    pred = forecast_multi_step(model, _last_window(), steps=10)
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()
    assert pred.dtype == np.float32


def test_forecast_multi_step_deterministic():
    model = build_gru_by_size("small", 24, 1, seed=42)
    win = _last_window()
    first = forecast_multi_step(model, win, steps=4)
    second = forecast_multi_step(model, win, steps=4)
    np.testing.assert_array_equal(first, second)


def test_forecast_multi_step_validates():
    model = build_gru_by_size("small", 24, 1, seed=42)
    with pytest.raises(ForecastError, match="steps"):
        forecast_multi_step(model, _last_window(), steps=0)
    with pytest.raises(ForecastError, match="window must have shape"):
        forecast_multi_step(model, np.zeros((8, 1)), steps=3)


def test_forecast_series_schema_single():
    model = build_gru_by_size("small", 24, 1, seed=42)
    df = _window_df()
    frame = forecast_series(model, df, steps=4)
    assert list(frame.columns) == FORECAST_COLUMNS
    assert len(frame) == 4
    assert frame["horizon_step"].tolist() == [1, 2, 3, 4]
    assert (frame["symbol_or_series_id"] == "series_single").all()
    assert (frame["model"] == model.name).all()
    assert frame["predicted_value"].dtype == np.float64
    expected = forecast_multi_step(model, _last_window(df), steps=4)
    np.testing.assert_allclose(frame["predicted_value"].to_numpy(), expected, atol=1e-6)


def test_forecast_series_timestamps_advance_hourly():
    model = build_gru_by_size("small", 24, 1, seed=42)
    df = _window_df()
    frame = forecast_series(model, df, steps=3)
    last = df["timestamp"].max()
    assert frame["timestamp"].iloc[0] == last + pd.Timedelta(hours=1)
    assert (frame["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert frame["timestamp"].dt.tz is not None


def test_forecast_series_freq_override():
    model = build_gru_by_size("small", 24, 1, seed=42)
    df = _window_df()
    frame = forecast_series(model, df, steps=3, freq="2h")
    assert (frame["timestamp"].diff().dropna() == pd.Timedelta(hours=2)).all()


def test_forecast_series_multi_series_fixture():
    model = build_gru_by_size("small", 24, 1, seed=42)
    df = CsvSource(MULTI).read()
    frame = forecast_series(model, df, steps=3)
    assert len(frame) == 6
    assert sorted(frame["symbol_or_series_id"].unique()) == ["A", "B"]
    assert frame.groupby("symbol_or_series_id").size().eq(3).all()


def test_forecast_series_irregular_frequency_fallback():
    model = build_gru_by_size("small", 24, 1, seed=42)
    df = _window_df().copy()
    jitter = pd.to_timedelta(np.where(np.arange(len(df)) == 20, 2, 0), unit="m")
    df["timestamp"] = df["timestamp"] + jitter
    frame = forecast_series(model, df, steps=2)
    assert len(frame) == 2
    assert frame["timestamp"].iloc[0] == df["timestamp"].max() + pd.Timedelta(minutes=60)


def test_forecast_series_invalid_df_raises():
    model = build_gru_by_size("small", 24, 1, seed=42)
    bad = _window_df().rename(columns={"value": "vals"})
    with pytest.raises(PipelineError):
        forecast_series(model, bad, steps=2)


def test_forecast_series_rejects_undeterminable_freq():
    model = build_gru_by_size("small", 1, 1, seed=42)
    df = _window_df().head(1).copy()
    with pytest.raises(ForecastError, match="freq"):
        forecast_series(model, df, steps=2)


def test_load_trained_model_independent_of_training_mode(tmp_path):
    x, y = _windowed_train_data()
    model = build_gru_by_size("small", 24, 1, seed=42)
    result = train(
        model,
        x,
        y,
        x_val=x,
        y_val=y,
        epochs=2,
        batch_size=32,
        mode="xla",
        checkpoint_dir=tmp_path,
        early_stopping_patience=0,
    )
    restored = load_trained_model("gru", "small", 24, 1, result.checkpoint_path, seed=42)
    win = _last_window()
    np.testing.assert_allclose(
        restored(win[None, ...]).numpy(),
        model(win[None, ...]).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_load_trained_model_lstm(tmp_path):
    x, y = _windowed_train_data()
    model = build_lstm_by_size("small", 24, 1, seed=7)
    result = train(
        model,
        x,
        y,
        x_val=x,
        y_val=y,
        epochs=2,
        batch_size=32,
        mode="eager",
        checkpoint_dir=tmp_path,
        early_stopping_patience=0,
    )
    restored = load_trained_model("lstm", "small", 24, 1, result.checkpoint_path, seed=7)
    assert restored.name == "lstm_small"


def test_load_trained_model_unknown_model_raises():
    with pytest.raises(ForecastError, match="unknown model"):
        load_trained_model("transformer", "small", 24, 1, "x.weights.h5")


def _windowed_train_data():
    df = CsvSource(MULTI).read()
    x, y = _windowed_from(df)
    return x, y


def _windowed_from(df, lookback=24, horizon=1):
    from kasoku.pipeline import window_series

    return window_series(df, lookback=lookback, horizon=horizon)
