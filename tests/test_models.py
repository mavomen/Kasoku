"""Unit tests for the GRU, LSTM, and TCN forecaster architectures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kasoku.data import CsvSource
from kasoku.models import (
    GRU_SIZES,
    LSTM_SIZES,
    TCN_SIZES,
    ModelError,
    build_gru,
    build_gru_by_size,
    build_lstm,
    build_lstm_by_size,
    build_tcn,
    build_tcn_by_size,
)
from kasoku.pipeline import window_series

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"


def _samples(count: int, lookback: int) -> np.ndarray:
    return np.random.default_rng(7).normal(size=(count, lookback, 1)).astype(np.float32)


def test_gru_small_forward_shape():
    model = build_gru_by_size("small", lookback=24, horizon=1)
    out = model(_samples(8, 24))
    assert out.shape == (8, 1)
    assert model.name == "gru_small"


def test_gru_medium_forward_shape():
    model = build_gru_by_size("medium", lookback=24, horizon=4)
    out = model(_samples(8, 24))
    assert out.shape == (8, 4)


def test_gru_core_factory_stacked_layers():
    model = build_gru(24, 2, hidden_units=16, num_layers=2, dropout=0.0)
    assert len(model.layers) >= 4
    out = model(_samples(8, 24))
    assert out.shape == (8, 2)


def test_gru_outputs_are_finite():
    model = build_gru_by_size("small", lookback=24, horizon=3)
    out = model(_samples(16, 24))
    assert np.isfinite(out.numpy()).all()


def test_gru_forward_pass_from_fixture():
    df = CsvSource(SINGLE).read()
    x, _ = window_series(df, lookback=24, horizon=1)
    assert x.shape == (72, 24, 1)
    model = build_gru_by_size("small", lookback=24, horizon=1)
    out = model(x)
    assert out.shape == (72, 1)
    assert np.isfinite(out.numpy()).all()


def test_gru_deterministic_with_fixed_seed():
    first = build_gru_by_size("small", lookback=24, horizon=1, seed=42)
    second = build_gru_by_size("small", lookback=24, horizon=1, seed=42)
    x = _samples(4, 24)
    np.testing.assert_allclose(first(x).numpy(), second(x).numpy(), atol=1e-6)


def test_gru_unknown_size_raises():
    with pytest.raises(ModelError, match="unknown gru size"):
        build_gru_by_size("huge", lookback=24, horizon=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0, "horizon": 1},
        {"lookback": 24, "horizon": 0},
        {"lookback": 24, "horizon": 1, "num_layers": 0},
        {"lookback": 24, "horizon": 1, "hidden_units": 0},
    ],
)
def test_gru_invalid_config_raises(kwargs):
    with pytest.raises(ModelError):
        build_gru(**kwargs)


def test_gru_sizes_registry():
    assert set(GRU_SIZES) == {"small", "medium"}
    for config in GRU_SIZES.values():
        assert config["hidden_units"] >= 1
        assert config["num_layers"] >= 1
        assert 0.0 <= config["dropout"] < 1.0
    assert GRU_SIZES["small"]["hidden_units"] < GRU_SIZES["medium"]["hidden_units"]


def test_lstm_small_forward_shape():
    model = build_lstm_by_size("small", lookback=24, horizon=1)
    out = model(_samples(8, 24))
    assert out.shape == (8, 1)
    assert model.name == "lstm_small"


def test_lstm_medium_forward_shape():
    model = build_lstm_by_size("medium", lookback=24, horizon=4)
    out = model(_samples(8, 24))
    assert out.shape == (8, 4)


def test_lstm_core_factory_stacked_layers():
    model = build_lstm(24, 2, hidden_units=16, num_layers=2, dropout=0.0)
    assert len(model.layers) >= 4
    out = model(_samples(8, 24))
    assert out.shape == (8, 2)


def test_lstm_outputs_are_finite():
    model = build_lstm_by_size("small", lookback=24, horizon=3)
    out = model(_samples(16, 24))
    assert np.isfinite(out.numpy()).all()


def test_lstm_forward_pass_from_fixture():
    df = CsvSource(SINGLE).read()
    x, _ = window_series(df, lookback=24, horizon=1)
    assert x.shape == (72, 24, 1)
    model = build_lstm_by_size("small", lookback=24, horizon=1)
    out = model(x)
    assert out.shape == (72, 1)
    assert np.isfinite(out.numpy()).all()


def test_lstm_deterministic_with_fixed_seed():
    first = build_lstm_by_size("small", lookback=24, horizon=1, seed=42)
    second = build_lstm_by_size("small", lookback=24, horizon=1, seed=42)
    x = _samples(4, 24)
    np.testing.assert_allclose(first(x).numpy(), second(x).numpy(), atol=1e-6)


def test_lstm_unknown_size_raises():
    with pytest.raises(ModelError, match="unknown lstm size"):
        build_lstm_by_size("huge", lookback=24, horizon=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0, "horizon": 1},
        {"lookback": 24, "horizon": 0},
        {"lookback": 24, "horizon": 1, "num_layers": 0},
        {"lookback": 24, "horizon": 1, "hidden_units": 0},
    ],
)
def test_lstm_invalid_config_raises(kwargs):
    with pytest.raises(ModelError):
        build_lstm(**kwargs)


def test_lstm_sizes_registry():
    assert set(LSTM_SIZES) == {"small", "medium"}
    for config in LSTM_SIZES.values():
        assert config["hidden_units"] >= 1
        assert config["num_layers"] >= 1
        assert 0.0 <= config["dropout"] < 1.0
    assert LSTM_SIZES["small"]["hidden_units"] < LSTM_SIZES["medium"]["hidden_units"]


def test_tcn_small_forward_shape():
    model = build_tcn_by_size("small", lookback=24, horizon=1)
    out = model(_samples(8, 24))
    assert out.shape == (8, 1)
    assert model.name == "tcn_small"


def test_tcn_medium_forward_shape():
    model = build_tcn_by_size("medium", lookback=24, horizon=4)
    out = model(_samples(8, 24))
    assert out.shape == (8, 4)


def test_tcn_core_factory_stacked_blocks():
    model = build_tcn(24, 2, filters=16, num_blocks=2, kernel_size=3, dropout=0.0)
    assert len(model.layers) >= 4
    out = model(_samples(8, 24))
    assert out.shape == (8, 2)


def test_tcn_outputs_are_finite():
    model = build_tcn_by_size("small", lookback=24, horizon=3)
    out = model(_samples(16, 24))
    assert np.isfinite(out.numpy()).all()


def test_tcn_forward_pass_from_fixture():
    df = CsvSource(SINGLE).read()
    x, _ = window_series(df, lookback=24, horizon=1)
    assert x.shape == (72, 24, 1)
    model = build_tcn_by_size("small", lookback=24, horizon=1)
    out = model(x)
    assert out.shape == (72, 1)
    assert np.isfinite(out.numpy()).all()


def test_tcn_deterministic_with_fixed_seed():
    first = build_tcn_by_size("small", lookback=24, horizon=1, seed=42)
    second = build_tcn_by_size("small", lookback=24, horizon=1, seed=42)
    x = _samples(4, 24)
    np.testing.assert_allclose(first(x).numpy(), second(x).numpy(), atol=1e-6)


def test_tcn_unknown_size_raises():
    with pytest.raises(ModelError, match="unknown tcn size"):
        build_tcn_by_size("huge", lookback=24, horizon=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0, "horizon": 1},
        {"lookback": 24, "horizon": 0},
        {"lookback": 24, "horizon": 1, "num_blocks": 0},
        {"lookback": 24, "horizon": 1, "filters": 0},
        {"lookback": 24, "horizon": 1, "kernel_size": 0},
        {"lookback": 24, "horizon": 1, "dilation_base": 0},
        {"lookback": 24, "horizon": 1, "dropout": 1.0},
    ],
)
def test_tcn_invalid_config_raises(kwargs):
    with pytest.raises(ModelError):
        build_tcn(**kwargs)


def test_tcn_sizes_registry():
    assert set(TCN_SIZES) == {"small", "medium"}
    for config in TCN_SIZES.values():
        assert config["filters"] >= 1
        assert config["num_blocks"] >= 1
        assert config["kernel_size"] >= 1
        assert config["dilation_base"] >= 1
        assert 0.0 <= config["dropout"] < 1.0
    assert TCN_SIZES["small"]["filters"] < TCN_SIZES["medium"]["filters"]
