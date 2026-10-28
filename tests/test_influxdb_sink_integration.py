"""Integration tests for the InfluxDB sink against a real InfluxDB 2.x server.

Spins up ``influxdb:2.7`` via docker compose, writes forecast and benchmark
frames through ``InfluxDBSink``, queries them back with Flux, and asserts the
tags/fields round-trip correctly. Skips when docker or ``influxdb-client`` is
unavailable.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

from kasoku.benchmark import BENCHMARK_COLUMNS
from kasoku.forecast import FORECAST_COLUMNS
from kasoku.sinks.influxdb import InfluxDBSink

COMPOSE_FILE = str(Path(__file__).resolve().parent.parent / "docker-compose.integration.yml")
INFLUX_URL_BASE = "http://localhost"
INFLUX_TOKEN = "kasoku-integration-test-token"
INFLUX_ORG = "kasoku-org"


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "ps"], capture_output=True, check=True, timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _influxdb_available() -> bool:
    try:
        import influxdb_client  # noqa: F401
        return True
    except ImportError:
        return False


def _mapped_port(compose_file: str) -> str | None:
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "port", "influxdb", "8086"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().rsplit(":", 1)[-1] or None


def _wait_for_health(url: str, timeout: float = 60.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _ensure_onboarded(url: str) -> None:
    """Complete InfluxDB onboarding via the REST API if it has not been done."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(url + "/api/v2/setup", timeout=5) as response:
            setup = json.loads(response.read().decode())
    except Exception:
        return
    if not setup.get("allowed", False):
        return
    payload = json.dumps(
        {
            "org": INFLUX_ORG,
            "bucket": "kasoku",
            "username": "kasoku",
            "password": "password",
            "token": INFLUX_TOKEN,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/api/v2/setup",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        response.read()


@pytest.fixture(scope="session")
def influxdb_connection():
    """Start an influxdb:2.7 container and yield connection parameters."""
    if not _docker_available():
        pytest.skip("docker is unavailable")
    if not _influxdb_available():
        pytest.skip("influxdb-client is not installed")

    try:
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
            capture_output=True,
            timeout=180,
            check=True,
        )
    except subprocess.SubprocessError:
        pytest.skip("could not start influxdb container")

    try:
        host_port = _mapped_port(COMPOSE_FILE)
        if not host_port:
            pytest.skip("could not determine influxdb host port")
        url = f"{INFLUX_URL_BASE}:{host_port}"
        if not _wait_for_health(url):
            pytest.skip("influxdb did not become ready")
        _ensure_onboarded(url)
        yield {"url": url, "token": INFLUX_TOKEN, "org": INFLUX_ORG}
    finally:
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
            capture_output=True,
            timeout=180,
        )


def _create_unique_bucket(connection: dict) -> str:
    from influxdb_client import InfluxDBClient

    client = InfluxDBClient(
        url=connection["url"], token=connection["token"], org=connection["org"]
    )
    try:
        orgs = client.organizations_api().find_organizations(org=connection["org"])
        if not orgs:
            raise RuntimeError(f"organization {connection['org']!r} not found")
        org = orgs[0]
        bucket_name = f"kasoku-it-{int(time.time() * 1000)}"
        client.buckets_api().create_bucket(
            bucket_name=bucket_name, org=org.id, retention_rules=[]
        )
        return bucket_name
    finally:
        client.close()


def _query_points(connection: dict, bucket: str, measurement: str):
    from influxdb_client import InfluxDBClient

    flux = (
        f'from(bucket: "{bucket}") '
        '|> range(start: 1970-01-01T00:00:00Z) '
        f'|> filter(fn: (r) => r._measurement == "{measurement}")'
    )
    client = InfluxDBClient(
        url=connection["url"], token=connection["token"], org=connection["org"]
    )
    try:
        tables = client.query_api().query(org=connection["org"], query=flux)
    finally:
        client.close()
    records = []
    for table in tables:
        records.extend(table.records)
    return records


def _query_points_retry(
    connection: dict, bucket: str, measurement: str, min_records: int, timeout: float = 20.0
):
    deadline = time.time() + timeout
    last = []
    while time.time() < deadline:
        last = _query_points(connection, bucket, measurement)
        if len(last) >= min_records:
            return last
        time.sleep(0.5)
    return last


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-05T00:00:00Z", "2024-01-05T01:00:00Z"]
            ),
            "symbol_or_series_id": ["series_single", "series_single"],
            "horizon_step": [1, 2],
            "predicted_value": [1.25, 1.5],
            "model": ["gru", "gru"],
        }
    )


def _benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "model": ["gru", "gru"],
            "model_size": ["small", "medium"],
            "mode": ["eager", "xla"],
            "phase": ["inference", "inference"],
            "wall_clock_seconds": [0.1, 0.2],
            "throughput_samples_per_sec": [160.0, 80.0],
            "compile_overhead_seconds": [float("nan"), 0.5],
            "peak_memory_mb": [1.0, 2.0],
        }
    )


def test_influx_sink_integration_forecast(influxdb_connection):
    bucket = _create_unique_bucket(influxdb_connection)
    sink = InfluxDBSink(
        url=influxdb_connection["url"],
        token=influxdb_connection["token"],
        org=influxdb_connection["org"],
        bucket=bucket,
    )
    sink.write_forecast(_forecast_frame())

    records = _query_points_retry(
        influxdb_connection, bucket, "kasoku_forecast", min_records=4
    )
    tags_seen = set()
    field_values: dict[str, list] = {}
    for record in records:
        values = record.values
        tags_seen.add((values["symbol_or_series_id"], values["model"]))
        field_values.setdefault(values["_field"], []).append(values["_value"])

    assert tags_seen == {("series_single", "gru")}
    assert sorted(field_values["horizon_step"]) == [1, 2]
    assert sorted(field_values["predicted_value"]) == [1.25, 1.5]


def test_influx_sink_integration_benchmark(influxdb_connection):
    bucket = _create_unique_bucket(influxdb_connection)
    sink = InfluxDBSink(
        url=influxdb_connection["url"],
        token=influxdb_connection["token"],
        org=influxdb_connection["org"],
        bucket=bucket,
    )
    sink.write_benchmark(_benchmark_frame())

    records = _query_points_retry(
        influxdb_connection, bucket, "kasoku_benchmark", min_records=7
    )
    by_point: dict[tuple, dict] = {}
    for record in records:
        values = record.values
        key = (
            values["run_id"],
            values["model"],
            values["model_size"],
            values["mode"],
            values["phase"],
        )
        by_point.setdefault(key, {})[values["_field"]] = values["_value"]

    eager = by_point[("run-1", "gru", "small", "eager", "inference")]
    xla = by_point[("run-1", "gru", "medium", "xla", "inference")]
    assert "compile_overhead_seconds" not in eager
    assert eager["wall_clock_seconds"] == 0.1
    assert eager["throughput_samples_per_sec"] == 160.0
    assert eager["peak_memory_mb"] == 1.0
    assert xla["compile_overhead_seconds"] == 0.5
    assert xla["wall_clock_seconds"] == 0.2
    assert xla["throughput_samples_per_sec"] == 80.0
    assert xla["peak_memory_mb"] == 2.0
