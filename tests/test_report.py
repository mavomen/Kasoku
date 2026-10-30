"""Unit tests for the kasoku.report module."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from kasoku.report import (
    BENCHMARK_COLUMNS,
    ReportError,
    build_html,
    compute_xla_speedup,
    generate_report,
    load_benchmark_data,
)
from kasoku.report.watcher import _current_mtime, _file_changed, _run_polling, watch_and_regenerate

BENCHMARK_CSV = (
    "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
    "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
    "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n"
    "run-1,gru,medium,eager,inference,3.0,8.0,,110.0\n"
    "run-1,gru,medium,xla,inference,1.5,16.0,0.3,115.0\n"
    "run-1,lstm,small,eager,inference,2.5,9.0,,95.0\n"
    "run-1,lstm,small,xla,inference,2.0,12.0,0.25,100.0\n"
    "run-1,gru,small,eager,train,5.0,4.0,0.1,120.0\n"
)


def _write_csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_load_benchmark_data_from_csv(tmp_path):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    frame = load_benchmark_data(path)
    assert list(frame.columns) == BENCHMARK_COLUMNS
    assert len(frame) == 7
    assert frame["model"].tolist() == ["gru", "gru", "gru", "gru", "lstm", "lstm", "gru"]


def test_load_benchmark_data_from_json(tmp_path):
    records = [
        {
            "run_id": "run-1",
            "model": "gru",
            "model_size": "small",
            "mode": "eager",
            "phase": "inference",
            "wall_clock_seconds": 2.0,
            "throughput_samples_per_sec": 10.0,
            "compile_overhead_seconds": None,
            "peak_memory_mb": 100.0,
        }
    ]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    frame = load_benchmark_data(str(path))
    assert len(frame) == 1
    assert frame["model"].iloc[0] == "gru"


def test_load_benchmark_data_unsupported_source(tmp_path):
    path = _write_csv(tmp_path, "bench.txt", "not a benchmark\n")
    with pytest.raises(ReportError, match="unsupported report source"):
        load_benchmark_data(str(path))


def test_load_benchmark_data_missing_columns(tmp_path):
    path = _write_csv(tmp_path, "bench.csv", "a,b,c\n1,2,3\n")
    with pytest.raises(ReportError, match="missing columns"):
        load_benchmark_data(str(path))


def test_compute_xla_speedup():
    frame = pd.read_csv(__import__("io").StringIO(BENCHMARK_CSV))
    speedups = compute_xla_speedup(frame)
    by_key = {(s["model"], s["model_size"]): s["speedup"] for s in speedups}
    assert by_key[("gru", "small")] == pytest.approx(2.0 / 1.0)
    assert by_key[("gru", "medium")] == pytest.approx(3.0 / 1.5)
    assert by_key[("lstm", "small")] == pytest.approx(2.5 / 2.0)


def test_compute_xla_speedup_skips_incomplete_cells():
    csv_incomplete = (
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        # no xla cell for gru/small -> skipped
        "run-1,gru,medium,xla,inference,1.5,16.0,0.3,115.0\n"
        # no eager cell for gru/medium -> skipped
    )
    frame = pd.read_csv(__import__("io").StringIO(csv_incomplete))
    assert compute_xla_speedup(frame) == []


def test_compute_xla_speedup_ignores_train_phase():
    csv_train_only = (
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,train,5.0,4.0,0.1,120.0\n"
        "run-1,gru,small,xla,train,4.0,5.0,0.1,125.0\n"
    )
    frame = pd.read_csv(__import__("io").StringIO(csv_train_only))
    assert compute_xla_speedup(frame) == []


def test_build_html_contains_charts_and_data(tmp_path):
    frame = pd.read_csv(__import__("io").StringIO(BENCHMARK_CSV))
    html = build_html(frame)
    assert "<!DOCTYPE html>" in html
    assert 'id="chart-speedup"' in html
    assert 'id="chart-compile"' in html
    assert 'id="chart-throughput"' in html
    assert 'id="chart-memory"' in html
    assert "cdn.jsdelivr.net/npm/chart.js@4" in html
    assert "htmx.org" in html
    assert "const BENCHMARK_DATA = [" in html
    assert "__BENCHMARK_DATA__" not in html  # placeholder replaced


def test_build_html_with_compare(tmp_path):
    frame = pd.read_csv(__import__("io").StringIO(BENCHMARK_CSV))
    compare = pd.read_csv(__import__("io").StringIO(BENCHMARK_CSV))
    html = build_html(frame, compare_frame=compare)
    assert "__COMPARE_DATA__" not in html  # placeholder replaced
    assert "const COMPARE_DATA = [" in html


def test_generate_report_writes_file(tmp_path):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    output = tmp_path / "report.html"
    result = generate_report(str(path), str(output))
    assert result == str(output)
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "<canvas id=\"chart-speedup\"" in content or 'id="chart-speedup"' in content


def test_generate_report_empty_data_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        "bench.csv",
        ",".join(BENCHMARK_COLUMNS) + "\n",
    )
    with pytest.raises(ReportError, match="empty"):
        generate_report(str(path), str(tmp_path / "report.html"))


def test_generate_report_compare_empty_raises(tmp_path):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    empty = _write_csv(tmp_path, "empty.csv", ",".join(BENCHMARK_COLUMNS) + "\n")
    with pytest.raises(ReportError, match="compare benchmark data is empty"):
        generate_report(str(path), str(tmp_path / "report.html"), compare=str(empty))


def test_report_error_is_config_error():
    assert issubclass(ReportError, ValueError)
    err = ReportError("boom")
    assert str(err) == "boom"


def test_current_mtime_returns_number(tmp_path):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    assert _current_mtime(path) > 0
    assert _current_mtime(str(tmp_path / "missing.csv")) is None


def test_file_changed_detects_mtime_change(tmp_path):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    assert _file_changed(str(path), None) is False
    assert _file_changed(str(path), _current_mtime(str(path))) is False


def test_run_polling_regenerates_on_change(tmp_path, monkeypatch):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    output = tmp_path / "report.html"
    calls = []

    def fake_generate(data, output_path, *, compare=None, config=None, time_range="-24h"):
        calls.append((data, output_path))
        Path(output_path).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr("kasoku.report.watcher.generate_report", fake_generate)
    sleep_calls = {"n": 0}

    def fake_sleep(interval):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr("kasoku.report.watcher.time.sleep", fake_sleep)
    flags = iter([False, True, False])
    monkeypatch.setattr(
        "kasoku.report.watcher._file_changed", lambda path, last_mtime: next(flags)
    )
    with pytest.raises(KeyboardInterrupt):
        _run_polling(str(path), str(output), None, None, "-24h", interval=1)
    # initial generation + one regeneration on change
    assert len(calls) == 2


def test_load_from_influxdb_requires_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "influxdb_client", None)
    with pytest.raises(ReportError, match="influxdb-client is required"):
        load_benchmark_data("http://localhost:8086")


class _FakeRecord:
    def __init__(self, values):
        self.values = values


class _FakeTable:
    def __init__(self, records):
        self.records = [ _FakeRecord(r) for r in records ]


class _FakeQueryApi:
    def __init__(self, records):
        self.records = records
    def query(self, query, org):
        return [ _FakeTable(self.records) ]


class _FakeClient:
    def __init__(self, url, token, org):
        self.kwargs = (url, token, org)
        self.records = []
    def query_api(self):
        return self
    def query(self, query, org):
        return [ _FakeTable(self.records) ]
    def close(self):
        pass


def test_load_from_influxdb_success(monkeypatch):
    records = [{
        "_time": "2024-01-01T00:00:00Z",
        "_measurement": "kasoku_benchmark",
        "run_id": "run-1",
        "model": "gru",
        "model_size": "small",
        "mode": "xla",
        "phase": "inference",
        "wall_clock_seconds": 1.0,
        "throughput_samples_per_sec": 20.0,
        "compile_overhead_seconds": 0.2,
        "peak_memory_mb": 105.0,
    }]
    client_instance = _FakeClient("http://localhost:8086", "token", "org")
    client_instance.records = records
    module = types.SimpleNamespace(InfluxDBClient=lambda url, token, org: client_instance)
    monkeypatch.setitem(sys.modules, "influxdb_client", module)
    frame = load_benchmark_data("http://localhost:8086")
    assert len(frame) == 1
    assert list(frame.columns) == BENCHMARK_COLUMNS
    assert frame["model"].iloc[0] == "gru"


def test_load_from_influxdb_missing_column(monkeypatch):
    records = [{
        "_time": "2024-01-01T00:00:00Z",
        "_measurement": "kasoku_benchmark",
        "run_id": "run-1",
        "model": "gru",
        "model_size": "small",
        "mode": "xla",
        "phase": "inference",
        "wall_clock_seconds": 1.0,
        # peak_memory_mb intentionally absent -> filled with None
    }]
    client_instance = _FakeClient("http://localhost:8086", "token", "org")
    client_instance.records = records
    module = types.SimpleNamespace(InfluxDBClient=lambda url, token, org: client_instance)
    monkeypatch.setitem(sys.modules, "influxdb_client", module)
    frame = load_benchmark_data("http://localhost:8086")
    assert len(frame) == 1
    assert frame["peak_memory_mb"].isna().iloc[0]


def test_load_from_influxdb_empty(monkeypatch):
    client_instance = _FakeClient("http://localhost:8086", "token", "org")
    client_instance.records = []
    module = types.SimpleNamespace(InfluxDBClient=lambda url, token, org: client_instance)
    monkeypatch.setitem(sys.modules, "influxdb_client", module)
    frame = load_benchmark_data("http://localhost:8086")
    assert frame.empty
    assert list(frame.columns) == BENCHMARK_COLUMNS


def test_watch_and_regenerate_watchdog_branch(monkeypatch, tmp_path):
    path = _write_csv(tmp_path, "bench.csv", BENCHMARK_CSV)
    output = tmp_path / "report.html"
    calls = []

    def fake_generate(data, output_path, *, compare=None, config=None, time_range="-24h"):
        calls.append((data, output_path))
        Path(output_path).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr("kasoku.report.watcher.generate_report", fake_generate)

    class FakeHandler:
        def __init__(self, callback):
            self.callback = callback

        def on_modified(self, event):
            if not event.is_directory:
                self.callback()

    class FakeObserver:
        def schedule(self, handler, path, recursive=False):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    fake_watchdog = types.SimpleNamespace(
        observers=types.SimpleNamespace(Observer=FakeObserver),
        events=types.SimpleNamespace(FileSystemEventHandler=FakeHandler),
    )
    monkeypatch.setitem(sys.modules, "watchdog", fake_watchdog)
    monkeypatch.setitem(sys.modules, "watchdog.observers", fake_watchdog.observers)
    monkeypatch.setitem(sys.modules, "watchdog.events", fake_watchdog.events)

    sleep_calls = {"n": 0}

    def fake_sleep(interval):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("kasoku.report.watcher.time.sleep", fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        watch_and_regenerate(str(path), str(output), interval=1)
    assert len(calls) >= 1


def test_build_html_escapes_and_injects_speedup():
    frame = pd.read_csv(__import__("io").StringIO(BENCHMARK_CSV))
    html = build_html(frame)
    assert '"speedup": 2.0' in html
    assert "__SPEEDUP_DATA__" not in html  # placeholder replaced
