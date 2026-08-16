"""Errors and shared validation for output sinks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class SinkError(ValueError):
    """Raised when a sink cannot write output or receives an invalid frame."""


class Sink(ABC):
    """Write forecast and benchmark frames to an output destination.

    Both write methods validate their frame against the corresponding output
    schema before writing, so a frame never reaches the destination with the
    wrong shape.
    """

    @abstractmethod
    def write_forecast(self, frame: pd.DataFrame) -> None:
        """Write a forecast frame (the ``FORECAST_COLUMNS`` schema)."""

    @abstractmethod
    def write_benchmark(self, frame: pd.DataFrame) -> None:
        """Write a benchmark frame (the ``BENCHMARK_COLUMNS`` schema)."""


def validate_schema(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise :class:`SinkError` unless ``frame`` matches ``columns`` exactly."""
    if set(frame.columns) != set(columns):
        raise SinkError(
            f"frame columns must match {list(columns)}, got {list(frame.columns)}"
        )
