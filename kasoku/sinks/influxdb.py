"""InfluxDB (v2) output sink for forecast and benchmark frames.

The ``influxdb-client`` package is imported lazily so it is never a hard
dependency: the sink only fails when actually used without the client
installed. Results are written to the ``kasoku_forecast`` and
``kasoku_benchmark`` measurements with string columns as tags and numeric
columns as fields, so results are dashboard-able without transformation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from kasoku.benchmark import BENCHMARK_COLUMNS
from kasoku.forecast import FORECAST_COLUMNS
from kasoku.sinks.base import Sink, SinkError, validate_schema

_BENCHMARK_TAGS = ("run_id", "model", "model_size", "mode", "phase")
_BENCHMARK_FIELDS = (
    "wall_clock_seconds",
    "throughput_samples_per_sec",
    "compile_overhead_seconds",
    "peak_memory_mb",
)


class InfluxDBSink(Sink):
    """Write output frames to an InfluxDB 2.x bucket."""

    def __init__(self, url: str, token: str, org: str, bucket: str) -> None:
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket

    def write_forecast(self, frame: pd.DataFrame) -> None:
        validate_schema(frame, FORECAST_COLUMNS)
        module = _influxdb()
        points = [_forecast_point(module.Point, row) for _, row in frame.iterrows()]
        self._write_points(module, points)

    def write_benchmark(self, frame: pd.DataFrame) -> None:
        validate_schema(frame, BENCHMARK_COLUMNS)
        module = _influxdb()
        now = datetime.now(timezone.utc)
        points = [_benchmark_point(module.Point, row, now) for _, row in frame.iterrows()]
        self._write_points(module, points)

    def _write_points(self, module, points: list) -> None:
        client = module.InfluxDBClient(url=self.url, token=self.token, org=self.org)
        try:
            client.write_api().write(bucket=self.bucket, org=self.org, record=points)
        finally:
            client.close()


def _forecast_point(point_class, row):
    point = point_class("kasoku_forecast").time(pd.Timestamp(row["timestamp"]).isoformat())
    point = point.tag("symbol_or_series_id", str(row["symbol_or_series_id"]))
    point = point.tag("model", str(row["model"]))
    point = point.field("horizon_step", int(row["horizon_step"]))
    point = point.field("predicted_value", float(row["predicted_value"]))
    return point


def _benchmark_point(point_class, row, now: datetime):
    point = point_class("kasoku_benchmark").time(now.isoformat())
    for tag in _BENCHMARK_TAGS:
        point = point.tag(tag, str(row[tag]))
    for field in _BENCHMARK_FIELDS:
        value = row[field]
        if not pd.isna(value):
            point = point.field(field, float(value))
    return point


def _influxdb():
    try:
        import influxdb_client
    except ImportError as exc:
        raise SinkError(
            "influxdb-client is required for the influxdb sink; "
            "install it with 'pip install influxdb-client'"
        ) from exc
    return influxdb_client
