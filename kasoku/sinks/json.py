"""JSON output sink for forecast and benchmark frames.

Writes the frame as a JSON array of record objects (ISO timestamps, ``null``
for missing values such as the eager-mode compile overhead).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kasoku.benchmark import BENCHMARK_COLUMNS
from kasoku.forecast import FORECAST_COLUMNS
from kasoku.sinks.base import Sink, validate_schema


class JsonSink(Sink):
    """Write output frames to a JSON file at ``path``."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def write_forecast(self, frame: pd.DataFrame) -> None:
        validate_schema(frame, FORECAST_COLUMNS)
        _write_json(frame, self.path)

    def write_benchmark(self, frame: pd.DataFrame) -> None:
        validate_schema(frame, BENCHMARK_COLUMNS)
        _write_json(frame, self.path)


def _write_json(frame: pd.DataFrame, path: str) -> None:
    frame.to_json(path, orient="records", date_format="iso", indent=2)
