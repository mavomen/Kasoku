"""Unit tests for the Prometheus Pushgateway client."""

from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

from kasoku.observability import (
    PushgatewayError,
    format_benchmark_metrics,
    push_benchmark_metrics,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "model": ["gru", "gru"],
            "model_size": ["small", "medium"],
            "mode": ["eager", "xla"],
            "phase": ["inference", "inference"],
            "wall_clock_seconds": [0.1, 0.25],
            "throughput_samples_per_sec": [160.0, 80.0],
            "compile_overhead_seconds": [float("nan"), 0.5],
            "peak_memory_mb": [1.0, 2.0],
        }
    )


class _CapturedResponse:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _CapturedSession:
    def __init__(self) -> None:
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data, headers, timeout))
        return _CapturedResponse()


def _install_fake_requests(monkeypatch, session):
    monkeypatch.setattr("kasoku.observability.pushgateway.requests.post", session.post)


def test_format_benchmark_metrics_contains_all_gauges():
    text = format_benchmark_metrics(_frame())
    for metric in ("wall_clock_seconds", "throughput_samples_per_sec", "peak_memory_mb"):
        assert f"# HELP kasoku_{metric}" in text
        assert f"# TYPE kasoku_{metric} gauge" in text
    assert 'kasoku_wall_clock_seconds{model="gru",model_size="small",mode="eager",phase="inference"} 0.1' in text


def test_format_benchmark_metrics_skips_nan_compile_overhead():
    text = format_benchmark_metrics(_frame())
    # eager row has NaN compile overhead -> no sample for that label set
    eager_sample = 'kasoku_compile_overhead_seconds{model="gru",model_size="small",mode="eager",phase="inference"}'
    assert eager_sample not in text
    # xla row reports 0.5
    assert 'kasoku_compile_overhead_seconds{model="gru",model_size="medium",mode="xla",phase="inference"} 0.5' in text


def test_format_benchmark_metrics_info_gauge_carries_run_id():
    text = format_benchmark_metrics(_frame())
    assert (
        'kasoku_benchmark_run{run_id="run-1",model="gru",model_size="small",mode="eager",phase="inference"} 1'
        in text
    )
    assert 'kasoku_benchmark_run{run_id="run-1",model="gru",model_size="medium",mode="xla",phase="inference"} 1' in text


def test_format_benchmark_metrics_escapes_label_values():
    frame = _frame()
    frame.loc[0, "model_size"] = 'small"weird\\size'
    text = format_benchmark_metrics(frame)
    assert 'model_size="small\\"weird\\\\size"' in text


def test_push_benchmark_metrics_posts_text_format(monkeypatch):
    session = _CapturedSession()
    _install_fake_requests(monkeypatch, session)
    push_benchmark_metrics(_frame(), "http://localhost:9091")
    assert len(session.calls) == 1
    url, data, headers, timeout = session.calls[0]
    assert url == "http://localhost:9091/metrics/job/kasoku_benchmark"
    assert headers == {"Content-Type": "text/plain"}
    assert "kasoku_wall_clock_seconds" in data
    assert "kasoku_benchmark_run" in data


def test_push_benchmark_metrics_http_error_raises(monkeypatch):
    session = _CapturedSession()
    session.responses = [_CapturedResponse(500, "internal error")]
    original = session.post

    def post(url, data=None, headers=None, timeout=None):
        call = original(url, data=data, headers=headers, timeout=timeout)
        return session.responses[0]

    session.post = post
    _install_fake_requests(monkeypatch, session)
    with pytest.raises(PushgatewayError, match="HTTP 500"):
        push_benchmark_metrics(_frame(), "http://localhost:9091")


def test_push_benchmark_metrics_request_error_raises(monkeypatch):
    import requests

    def post(url, data=None, headers=None, timeout=None):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("kasoku.observability.pushgateway.requests.post", post)
    with pytest.raises(PushgatewayError, match="refused"):
        push_benchmark_metrics(_frame(), "http://localhost:9091")
