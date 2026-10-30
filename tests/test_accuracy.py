"""Unit tests for forecast accuracy evaluation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kasoku.accuracy import (
    AccuracyError,
    AccuracyResult,
    compute_mae,
    compute_mape,
    evaluate_accuracy,
    save_accuracy_json,
)
from kasoku.config import AccuracyConfig, ConfigError, parse_config


class _FakeModel:
    def __init__(self, lookback: int = 8, horizon: int = 1):
        self.input_shape = (None, lookback)
        self.output_shape = (None, horizon)
        self.name = "gru"


def _series_df(length: int = 20, symbol: str = "s1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=length, freq="h"),
            "symbol_or_series_id": [symbol] * length,
            "value": np.arange(length, dtype=np.float64),
        }
    )


def test_compute_mae():
    predictions = np.array([1.0, 2.0, 3.0])
    actuals = np.array([1.0, 3.0, 2.0])
    assert compute_mae(predictions, actuals) == pytest.approx((0 + 1 + 1) / 3.0)


def test_compute_mape():
    predictions = np.array([1.0, 2.0, 3.0])
    actuals = np.array([1.0, 4.0, 6.0])
    expected = np.mean(np.abs((np.array([1.0, 4.0, 6.0]) - predictions) / (np.array([1.0, 4.0, 6.0]) + 1e-8)))
    assert compute_mape(predictions, actuals) == pytest.approx(expected)


def test_evaluate_accuracy_all_pass(monkeypatch):
    def fake_multi_step(model, window, steps, *, mode):
        return np.arange(16, 20, dtype=np.float64)

    monkeypatch.setattr("kasoku.accuracy.forecast_multi_step", fake_multi_step)

    df = _series_df(length=20)
    model = _FakeModel()
    results = evaluate_accuracy(model, df, horizon=4, mode="eager", config=AccuracyConfig())

    assert results["passed"] is True
    assert results["mae"]["enabled"] is True
    assert results["mae"]["value"] == 0.0
    assert results["mae"]["threshold"] == 2.0
    assert results["mae"]["passed"] is True
    assert results["mae"]["per_series"] == {"s1": 0.0}
    assert results["mape"]["passed"] is True


def test_evaluate_accuracy_fail(monkeypatch):
    def fake_multi_step(model, window, steps, *, mode):
        return np.full(steps, 100.0, dtype=np.float64)

    monkeypatch.setattr("kasoku.accuracy.forecast_multi_step", fake_multi_step)

    df = _series_df(length=20)
    results = evaluate_accuracy(
        model=_FakeModel(),
        df=df,
        horizon=4,
        mode="eager",
        config=AccuracyConfig(),
    )

    assert results["passed"] is False
    assert results["mae"]["passed"] is False
    assert results["mae"]["per_series"] == {"s1": float(np.mean(np.abs(np.full(4, 100.0) - np.arange(16, 20))))}


def test_evaluate_accuracy_disabled_metric_skipped(monkeypatch):
    monkeypatch.setattr(
        "kasoku.accuracy.forecast_multi_step",
        lambda model, window, steps, *, mode: np.full(steps, 100.0, dtype=np.float64),
    )
    config = AccuracyConfig()
    config.mae.enabled = False
    config.mape.enabled = False

    df = _series_df(length=20)
    results = evaluate_accuracy(_FakeModel(), df, horizon=4, mode="eager", config=config)

    assert results["passed"] is True
    assert results["mae"]["enabled"] is False
    assert results["mae"]["value"] is None
    assert results["mae"]["passed"] is True
    assert results["mae"]["per_series"] == {}


def test_evaluate_accuracy_short_series_skipped_not_failure(monkeypatch):
    seen = []

    def fake_multi_step(model, window, steps, *, mode):
        seen.append(True)
        return np.full(steps, 1.0, dtype=np.float64)

    monkeypatch.setattr("kasoku.accuracy.forecast_multi_step", fake_multi_step)

    # series length == horizon: cannot evaluate
    df = _series_df(length=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = evaluate_accuracy(_FakeModel(), df, horizon=4, mode="eager", config=AccuracyConfig())

    assert results["passed"] is True
    assert results["mae"]["value"] is None
    assert seen == []
    assert any("length 4 <= horizon 4" in str(w.message) for w in caught)


def test_evaluate_accuracy_multi_series_aggregation(monkeypatch):
    calls = []

    def fake_multi_step(model, window, steps, *, mode):
        calls.append(len(window))
        return np.ones(steps, dtype=np.float64)

    monkeypatch.setattr("kasoku.accuracy.forecast_multi_step", fake_multi_step)

    df = pd.concat(
        [
            _series_df(length=20, symbol="a"),
            _series_df(length=20, symbol="b"),
        ],
        ignore_index=True,
    )
    config = AccuracyConfig()
    config.mae.threshold = 1000.0
    config.mape.threshold = 1000.0
    results = evaluate_accuracy(_FakeModel(), df, horizon=4, mode="eager", config=config)

    assert results["passed"] is True
    assert set(results["mae"]["per_series"]) == {"a", "b"}
    assert results["mae"]["per_series"]["a"] == results["mae"]["per_series"]["b"]
    # mean of identical per-series values equals that value
    assert results["mae"]["value"] == results["mae"]["per_series"]["a"]
    assert len(calls) == 2


def test_evaluate_accuracy_missing_columns_raises():
    df = pd.DataFrame({"timestamp": [1], "symbol_or_series_id": ["a"]})
    with pytest.raises(AccuracyError, match="invalid forecast frame"):
        evaluate_accuracy(_FakeModel(), df, horizon=4, mode="eager", config=AccuracyConfig())


def test_evaluate_accuracy_invalid_horizon_raises():
    df = _series_df(length=20)
    with pytest.raises(AccuracyError, match="horizon must be >= 1"):
        evaluate_accuracy(_FakeModel(), df, horizon=0, mode="eager", config=AccuracyConfig())


def test_accuracy_config_defaults():
    config = parse_config({})
    assert config.accuracy.mae.enabled is True
    assert config.accuracy.mae.threshold == 2.0
    assert config.accuracy.mape.enabled is True
    assert config.accuracy.mape.threshold == 0.15


def test_accuracy_config_from_toml(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[accuracy]
mae = { enabled = false, threshold = 5.0 }
mape = { enabled = true, threshold = 0.05 }
""",
        encoding="utf-8",
    )
    from kasoku.config import load_config

    config = load_config(cfg)
    assert config.accuracy.mae.enabled is False
    assert config.accuracy.mae.threshold == 5.0
    assert config.accuracy.mape.enabled is True
    assert config.accuracy.mape.threshold == 0.05


@pytest.mark.parametrize("threshold", [0, -1.0])
def test_accuracy_invalid_threshold_raises(threshold):
    with pytest.raises(ConfigError):
        parse_config({"accuracy": {"mae": {"enabled": True, "threshold": threshold}}})


def test_accuracy_non_table_section_raises():
    with pytest.raises(ConfigError):
        parse_config({"accuracy": "fast"})


def test_save_accuracy_json_writes_schema(tmp_path):
    results = {
        "passed": True,
        "mae": {
            "value": 1.23,
            "threshold": 2.0,
            "passed": True,
            "enabled": True,
            "per_series": {"series_a": 1.1, "series_b": 1.4},
        },
        "mape": {
            "value": 0.08,
            "threshold": 0.15,
            "passed": True,
            "enabled": True,
            "per_series": {"series_a": 0.07, "series_b": 0.11},
        },
    }
    path = tmp_path / "accuracy.json"
    save_accuracy_json(results, str(path))
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == results
    assert written["passed"] is True
    assert written["mae"]["per_series"]["series_a"] == 1.1


def test_save_accuracy_json_none_value(tmp_path):
    results = {
        "passed": True,
        "mae": {
            "value": None,
            "threshold": 2.0,
            "passed": True,
            "enabled": True,
            "per_series": {},
        },
        "mape": {
            "value": None,
            "threshold": 0.15,
            "passed": True,
            "enabled": True,
            "per_series": {},
        },
    }
    path = tmp_path / "accuracy.json"
    save_accuracy_json(results, path)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["mae"]["value"] is None
    assert written["mape"]["value"] is None
