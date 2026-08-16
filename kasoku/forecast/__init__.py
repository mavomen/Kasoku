"""Forecast inference: point and multi-step predictions from trained models."""

from kasoku.forecast.base import ForecastError
from kasoku.forecast.forecaster import (
    FORECAST_COLUMNS,
    forecast_multi_step,
    forecast_point,
    forecast_series,
    load_trained_model,
)

__all__ = [
    "FORECAST_COLUMNS",
    "ForecastError",
    "forecast_multi_step",
    "forecast_point",
    "forecast_series",
    "load_trained_model",
]
