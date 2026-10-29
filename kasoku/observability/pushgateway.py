"""Pushgateway client for benchmark metrics.

Pushes benchmark result frames to a Prometheus Pushgateway in the Prometheus
text exposition format, so Prometheus can scrape them and evaluate alert
rules against the latest push.

Time-series gauges are keyed by ``model, model_size, mode, phase`` (no
``run_id`` to avoid label cardinality explosion); ``run_id`` appears only on
the ``kasoku_benchmark_run`` info gauge, per the observability plan.
"""

from __future__ import annotations

import pandas as pd
import requests

BENCHMARK_GAUGES = (
    "wall_clock_seconds",
    "throughput_samples_per_sec",
    "compile_overhead_seconds",
    "peak_memory_mb",
)

BENCHMARK_LABELS = ("model", "model_size", "mode", "phase")

PUSHGATEWAY_PATH = "/metrics/job/kasoku_benchmark"


class PushgatewayError(ValueError):
    """Raised when benchmark metrics cannot be pushed to the Pushgateway."""


def _escape_label(value: str) -> str:
    """Escape a label value for the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _label_set(labels: dict[str, str]) -> str:
    return ",".join(f'{key}="{_escape_label(value)}"' for key, value in labels.items())


def format_benchmark_metrics(frame: pd.DataFrame) -> str:
    """Render a benchmark results frame as Prometheus text-format metrics.

    Each row emits one gauge sample per numeric metric plus one ``info``
    sample carrying the run metadata (including ``run_id``).
    """
    lines: list[str] = []
    for metric in BENCHMARK_GAUGES:
        lines.append(f"# HELP kasoku_{metric} Benchmark {metric}")
        lines.append(f"# TYPE kasoku_{metric} gauge")
        for _, row in frame.iterrows():
            value = row[metric]
            if pd.isna(value):
                continue
            labels = _label_set({label: str(row[label]) for label in BENCHMARK_LABELS})
            lines.append(f"kasoku_{metric}{{{labels}}} {float(value)}")
    lines.append("# HELP kasoku_benchmark_run Benchmark run metadata (gauge)")
    lines.append("# TYPE kasoku_benchmark_run gauge")
    for _, row in frame.iterrows():
        labels = _label_set(
            {
                "run_id": str(row["run_id"]),
                **{label: str(row[label]) for label in BENCHMARK_LABELS},
            }
        )
        lines.append(f"kasoku_benchmark_run{{{labels}}} 1")
    return "\n".join(lines) + "\n"


def push_benchmark_metrics(frame, push_gateway_url: str, timeout: float = 30.0) -> None:
    """POST benchmark metrics to the Pushgateway ``/metrics/job/kasoku_benchmark`` endpoint.

    Re-pushing replaces same-labeled metrics, so the gateway always reflects
    the most recent push.
    """
    url = push_gateway_url.rstrip("/") + PUSHGATEWAY_PATH
    body = format_benchmark_metrics(frame)
    try:
        response = requests.post(url, data=body, headers={"Content-Type": "text/plain"}, timeout=timeout)
    except requests.RequestException as exc:
        raise PushgatewayError(f"failed to push metrics to {url}: {exc}") from exc
    if response.status_code >= 400:
        raise PushgatewayError(
            f"Pushgateway returned HTTP {response.status_code} for {url}: {response.text.strip()}"
        )


__all__ = [
    "BENCHMARK_GAUGES",
    "BENCHMARK_LABELS",
    "PushgatewayError",
    "format_benchmark_metrics",
    "push_benchmark_metrics",
]
