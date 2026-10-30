"""Load benchmark data from CSV, JSON, or InfluxDB for reporting.

Input sources mirror the benchmark output schema (see
:data:`kasoku.benchmark.runner.BENCHMARK_COLUMNS`). CSV and JSON are the
default sinks; InfluxDB is optional and read lazily so
``influxdb-client`` is never a hard dependency.

Also loads forecast accuracy JSON (written by ``kasoku accuracy check
--output``) for use in the accuracy report dashboard.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from kasoku.config import Config
from kasoku.report.errors import ReportError

BENCHMARK_COLUMNS = [
    "run_id",
    "model",
    "model_size",
    "mode",
    "phase",
    "wall_clock_seconds",
    "throughput_samples_per_sec",
    "compile_overhead_seconds",
    "peak_memory_mb",
]

ACCURACY_METRICS = ("mae", "mape")
ACCURACY_METRIC_KEYS = ("value", "threshold", "passed", "enabled", "per_series")


def load_benchmark_data(
    source: str,
    config: Config | None = None,
    time_range: str = "-24h",
) -> pd.DataFrame:
    """Load benchmark data from CSV, JSON, or InfluxDB.

    Args:
        source: Path to a ``.csv``/``.json`` file, or an InfluxDB URL
            (``http://...``).
        config: Optional config for InfluxDB credentials.
        time_range: InfluxDB Flux range, e.g. ``-1h`` or ``-24h``.

    Returns:
        DataFrame with :data:`BENCHMARK_COLUMNS`.

    Raises:
        ReportError: if the source is unsupported or unreadable.
    """
    if source.endswith(".csv"):
        frame = pd.read_csv(source)
    elif source.endswith(".json"):
        frame = pd.read_json(source)
    elif source.startswith(("http://", "https://")):
        frame = _load_from_influxdb(source, config, time_range)
    else:
        raise ReportError(f"unsupported report source: {source!r}")
    _validate_columns(frame)
    return frame


def load_accuracy_data(source: str) -> dict:
    """Load and validate accuracy results JSON written by ``accuracy check --output``.

    Args:
        source: Path to a ``.json`` file produced by ``kasoku accuracy check --output``.

    Returns:
        The parsed accuracy dict with ``passed``, ``mae``, and ``mape`` keys,
        each metric containing ``value``, ``threshold``, ``passed``, ``enabled``,
        and ``per_series``.

    Raises:
        ReportError: if the source is not a ``.json`` file, cannot be read, or
            does not match the expected accuracy schema.
    """
    if not source.endswith(".json"):
        raise ReportError(f"unsupported accuracy source: {source!r} (expected a .json file)")
    try:
        with open(source, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read accuracy data from {source!r}: {exc}") from exc
    _validate_accuracy_schema(data, source)
    return data


def _validate_accuracy_schema(data: dict, source: str) -> None:
    if not isinstance(data, dict):
        raise ReportError(f"accuracy data from {source!r} must be a JSON object")
    for key in ("passed", *ACCURACY_METRICS):
        if key not in data:
            raise ReportError(f"accuracy data from {source!r} missing key: {key}")
    if not isinstance(data["passed"], bool):
        raise ReportError(f"accuracy data from {source!r}: 'passed' must be a boolean")
    for metric in ACCURACY_METRICS:
        entry = data[metric]
        if not isinstance(entry, dict):
            raise ReportError(f"accuracy data from {source!r}: '{metric}' must be an object")
        for metric_key in ACCURACY_METRIC_KEYS:
            if metric_key not in entry:
                raise ReportError(
                    f"accuracy data from {source!r}: '{metric}' missing key: {metric_key}"
                )
        if not isinstance(entry["per_series"], dict):
            raise ReportError(
                f"accuracy data from {source!r}: '{metric}.per_series' must be an object"
            )


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in BENCHMARK_COLUMNS if column not in frame.columns]
    if missing:
        raise ReportError(f"benchmark data missing columns: {missing}")


def _load_from_influxdb(url: str, config: Config | None, time_range: str) -> pd.DataFrame:
    try:
        from influxdb_client import InfluxDBClient
    except ImportError as exc:
        raise ReportError(
            "influxdb-client is required for InfluxDB report sources; "
            "install it with 'pip install influxdb-client'"
        ) from exc
    influx = config.sink.influxdb if config is not None else None
    token = influx.token if influx is not None else ""
    org = influx.org if influx is not None else ""
    bucket = influx.bucket if influx is not None else "kasoku"
    query = (
        f'from(bucket: "{bucket}") '
        f'|> range(start: {time_range}) '
        f'|> filter(fn: (r) => r._measurement == "kasoku_benchmark") '
        f'|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )
    client = InfluxDBClient(url=url, token=token, org=org)
    try:
        tables = client.query_api().query(query=query, org=org)
    finally:
        client.close()
    records: list[dict] = []
    for table in tables:
        for record in table.records:
            records.append(record.values)
    if not records:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)
    frame = pd.DataFrame(records)
    for column in BENCHMARK_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[BENCHMARK_COLUMNS]
