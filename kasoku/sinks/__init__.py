"""Output sinks: CSV, JSON, and optional InfluxDB, selected from config.

Sink selection is fully config-driven via :class:`kasoku.config.SinkConfig`:
``create_sink`` returns the sink for the configured type (csv | json |
influxdb), and the ``write_forecast`` / ``write_benchmark`` helpers validate
and route a frame in one call. CSV/JSON sinks take their output path at call
time (the CLI ``--output`` flag), while the InfluxDB sink reads its connection
details from config. InfluxDB is never a hard dependency -- the client is
imported lazily inside :class:`kasoku.sinks.influxdb.InfluxDBSink`.
"""

from __future__ import annotations

import pandas as pd

from kasoku.config import SinkConfig
from kasoku.sinks.base import Sink, SinkError
from kasoku.sinks.csv import CsvSink
from kasoku.sinks.influxdb import InfluxDBSink
from kasoku.sinks.json import JsonSink

__all__ = [
    "CsvSink",
    "InfluxDBSink",
    "JsonSink",
    "Sink",
    "SinkError",
    "create_sink",
    "write_benchmark",
    "write_forecast",
]

_SINK_TYPES = ("csv", "json", "influxdb")


def create_sink(config: SinkConfig, output: str | None = None) -> Sink:
    """Build the sink configured in ``config`` (csv | json | influxdb).

    ``output`` is the destination path required by the csv/json sinks; the
    influxdb sink ignores it and uses its config connection details instead.
    """
    if config.type == "csv":
        return _path_sink(config, output, CsvSink)
    if config.type == "json":
        return _path_sink(config, output, JsonSink)
    if config.type == "influxdb":
        return InfluxDBSink(
            url=config.influxdb.url,
            token=config.influxdb.token,
            org=config.influxdb.org,
            bucket=config.influxdb.bucket,
        )
    raise SinkError(
        f"unknown sink type {config.type!r}; expected one of {list(_SINK_TYPES)}"
    )


def write_forecast(frame: pd.DataFrame, config: SinkConfig, output: str | None = None) -> None:
    """Validate and write a forecast frame through the configured sink."""
    create_sink(config, output).write_forecast(frame)


def write_benchmark(frame: pd.DataFrame, config: SinkConfig, output: str | None = None) -> None:
    """Validate and write a benchmark frame through the configured sink."""
    create_sink(config, output).write_benchmark(frame)


def _path_sink(config: SinkConfig, output: str | None, sink_class) -> Sink:
    if output is None:
        raise SinkError(f"sink type {config.type!r} requires an output path")
    return sink_class(output)
