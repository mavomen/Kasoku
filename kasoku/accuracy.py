"""Forecast accuracy evaluation (MAE/MAPE) for Kasoku.

Evaluates a trained model against held-out ground truth by splitting each
series at ``end - horizon``: the preceding data is used as the forecast
history and the final ``horizon`` values serve as ground truth. MAE and MAPE
are aggregated across all series (mean of per-series values) and compared
against configurable thresholds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from kasoku.config import AccuracyConfig, ConfigError
from kasoku.forecast import forecast_multi_step
from kasoku.pipeline import PipelineError, validate_canonical

REQUIRED_FORECAST_COLUMNS = {"timestamp", "symbol_or_series_id", "horizon_step", "predicted_value", "model"}


class AccuracyError(ConfigError):
    """Raised when forecast accuracy evaluation fails."""


@dataclass
class AccuracyResult:
    """Per-metric accuracy result.

    ``value`` is the mean across evaluable series (``None`` when nothing was
    evaluable or the metric is disabled). ``passed`` is ``True`` when the
    metric is disabled, when nothing was evaluable, or when ``value`` is at or
    below the threshold. ``per_series`` maps each series id to its value.
    """

    value: float | None
    threshold: float
    passed: bool
    enabled: bool
    per_series: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "enabled": self.enabled,
            "per_series": self.per_series,
        }


def compute_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - actuals)))


def compute_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Adds epsilon to avoid division by zero."""
    epsilon = 1e-8
    return float(np.mean(np.abs((actuals - predictions) / (actuals + epsilon))))


def _empty_metric(enabled: bool, threshold: float) -> dict:
    return {
        "value": None,
        "threshold": threshold,
        "passed": True,
        "enabled": enabled,
        "per_series": {},
    }


def evaluate_accuracy(
    model: tf.keras.Model,
    df: pd.DataFrame,
    horizon: int,
    mode: str,
    config: AccuracyConfig,
) -> dict[str, dict]:
    """Evaluate forecast accuracy for each series in ``df``.

    Each series is split at ``len(series) - horizon``: the preceding data is
    used as history and the final ``horizon`` values are the ground truth. The
    model forecasts ``horizon`` steps ahead of the history and predictions are
    compared to the actuals. Series that are too short to evaluate are skipped
    with a warning (never a failure).

    Returns a dict::

        {
            "passed": bool,
            "mae": {"value", "threshold", "passed", "enabled", "per_series"},
            "mape": {"value", "threshold", "passed", "enabled", "per_series"},
        }

    ``passed`` is True only when every enabled metric passes.
    """
    try:
        validate_canonical(df)
    except PipelineError as exc:
        raise AccuracyError(f"invalid forecast frame: {exc}") from exc

    if horizon < 1:
        raise AccuracyError(f"horizon must be >= 1, got {horizon}")
    results: dict[str, dict] = {
        "passed": True,
        "mae": _empty_metric(config.mae.enabled, config.mae.threshold),
        "mape": _empty_metric(config.mape.enabled, config.mape.threshold),
    }

    lookback = int(model.input_shape[1])
    mae_values: dict[str, float] = {}
    mape_values: dict[str, float] = {}

    for symbol, group in df.groupby("symbol_or_series_id", sort=True):
        group = group.sort_values("timestamp")
        values = group["value"].to_numpy(dtype=np.float64)

        if len(values) <= horizon:
            warnings.warn(
                f"series {symbol!r}: length {len(values)} <= horizon {horizon}; "
                "skipping accuracy evaluation",
                stacklevel=2,
            )
            continue
        if len(values) < lookback + horizon:
            warnings.warn(
                f"series {symbol!r}: length {len(values)} < lookback + horizon "
                f"({lookback} + {horizon}); skipping accuracy evaluation",
                stacklevel=2,
            )
            continue

        history = values[: len(values) - horizon]
        actuals = values[len(values) - horizon :]
        window = history[-lookback:, None]
        predictions = forecast_multi_step(model, window, horizon, mode=mode)

        if config.mae.enabled:
            mae_values[symbol] = compute_mae(predictions, actuals)
        if config.mape.enabled:
            mape_values[symbol] = compute_mape(predictions, actuals)

    for metric_name, per_series, target_cfg in (
        ("mae", mae_values, config.mae),
        ("mape", mape_values, config.mape),
    ):
        if not target_cfg.enabled:
            continue
        if not per_series:
            # nothing evaluable: not a failure
            results[metric_name] = _empty_metric(True, target_cfg.threshold)
            continue
        value = float(np.mean(list(per_series.values())))
        passed = value <= target_cfg.threshold
        results[metric_name] = {
            "value": value,
            "threshold": target_cfg.threshold,
            "passed": passed,
            "enabled": True,
            "per_series": per_series,
        }
        if not passed:
            results["passed"] = False

    return results


__all__ = [
    "AccuracyError",
    "AccuracyResult",
    "REQUIRED_FORECAST_COLUMNS",
    "compute_mae",
    "compute_mape",
    "evaluate_accuracy",
]
