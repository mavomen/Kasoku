"""Observability smoke test against a local Prometheus + Pushgateway + Grafana stack.

Spins up ``docker-compose.observability.yml``, runs ``kasoku benchmark --push-gateway``
against a fixture CSV, verifies Prometheus scraped the metrics and Grafana loads,
then simulates an XLA regression and asserts the critical Prometheus alert fires.
Skips when docker is unavailable.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest
import requests

from kasoku.benchmark import BENCHMARK_COLUMNS

COMPOSE_FILE = str(Path(__file__).resolve().parent.parent / "docker-compose.observability.yml")
PUSHGATEWAY_URL = "http://localhost:9091"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "ps"], capture_output=True, check=True, timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _wait_for(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                return
        except Exception as exc:
            last = exc
        time.sleep(1)
    raise AssertionError(f"service {url} did not become ready: {last}")


def _wait_until(condition, timeout: float = 30.0, interval: float = 1.5) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            if condition():
                return
        except Exception as exc:
            last = exc
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s: {last}")


@pytest.fixture(scope="session")
def observability_stack():
    """Start the observability compose stack and tear it down after the session."""
    if not _docker_available():
        pytest.skip("docker is unavailable")
    try:
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
            capture_output=True,
            timeout=180,
            check=True,
        )
    except subprocess.SubprocessError:
        pytest.skip("could not start observability stack")

    try:
        _wait_for(PUSHGATEWAY_URL + "/")
        _wait_for(PROMETHEUS_URL + "/-/healthy")
        _wait_for(GRAFANA_URL + "/api/health")
        yield
    finally:
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
            capture_output=True,
            timeout=180,
        )


def test_observability_smoke_end_to_end(observability_stack):
    subprocess.run(
        [
            "uv", "run", "kasoku", "benchmark",
            "--data", str(SINGLE),
            "--sizes", "small",
            "--modes", "eager,tf_function,xla",
            "--phases", "train,inference",
            "--output", "/tmp/kasoku-smoke-benchmark.csv",
            "--push-gateway", PUSHGATEWAY_URL,
        ],
        check=True,
        timeout=300,
    )

    text = requests.get(PUSHGATEWAY_URL + "/metrics", timeout=10).text
    assert "kasoku_wall_clock_seconds" in text
    assert "kasoku_throughput_samples_per_sec" in text
    assert "kasoku_compile_overhead_seconds" in text
    assert "kasoku_peak_memory_mb" in text
    assert "kasoku_benchmark_run" in text

    def metrics_scraped() -> bool:
        response = requests.get(
            PROMETHEUS_URL + "/api/v1/query",
            params={"query": "kasoku_wall_clock_seconds"},
            timeout=10,
        )
        return len(response.json()["data"]["result"]) > 0

    _wait_until(metrics_scraped, timeout=30)

    response = requests.get(GRAFANA_URL + "/api/health", timeout=10)
    assert response.status_code == 200


def test_observability_xla_regression_alert_fires(observability_stack):
    frame = pd.DataFrame(
        {
            "run_id": ["run-regress", "run-regress"],
            "model": ["gru", "gru"],
            "model_size": ["small", "small"],
            "mode": ["eager", "xla"],
            "phase": ["inference", "inference"],
            "wall_clock_seconds": [1.0, 2.0],
            "throughput_samples_per_sec": [10.0, 5.0],
            "compile_overhead_seconds": [float("nan"), 0.1],
            "peak_memory_mb": [100.0, 100.0],
        }
    )
    assert list(frame.columns) == BENCHMARK_COLUMNS

    from kasoku.observability import push_benchmark_metrics

    push_benchmark_metrics(frame, PUSHGATEWAY_URL)

    def critical_alert_firing() -> bool:
        response = requests.get(PROMETHEUS_URL + "/api/v1/alerts", timeout=10)
        alerts = response.json()["data"]["alerts"]
        return any(
            alert["labels"]["alertname"] == "KasokuXlaRegressionCritical"
            and alert["state"] == "firing"
            for alert in alerts
        )

    _wait_until(critical_alert_firing, timeout=45)
