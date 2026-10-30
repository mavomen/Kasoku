"""Load benchmark data from CSV, JSON, or InfluxDB for reporting.

Input sources mirror the benchmark output schema (see
:data:`kasoku.benchmark.runner.BENCHMARK_COLUMNS`). CSV and JSON are the
default sinks; InfluxDB is optional and read lazily so
``influxdb-client`` is never a hard dependency.
"""

from __future__ import annotations

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
