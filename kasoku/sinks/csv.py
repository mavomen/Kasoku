"""CSV output sink for forecast and benchmark frames.

Writes the frame verbatim with ``to_csv`` (index omitted), so timestamps are
ISO strings and missing values such as the eager-mode compile overhead are
empty cells that round-trip as NaN.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kasoku.benchmark import BENCHMARK_COLUMNS
from kasoku.forecast import FORECAST_COLUMNS
from kasoku.sinks.base import Sink, validate_schema


class CsvSink(Sink):
    """Write output frames to a CSV file at ``path``."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def write_forecast(self, frame: pd.DataFrame) -> None:
        validate_schema(frame, FORECAST_COLUMNS)
        _write_csv(frame, self.path)

    def write_benchmark(self, frame: pd.DataFrame) -> None:
        validate_schema(frame, BENCHMARK_COLUMNS)
        _write_csv(frame, self.path)


def _write_csv(frame: pd.DataFrame, path: str) -> None:
    frame.to_csv(path, index=False)
