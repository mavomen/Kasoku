"""Unit tests for the CSV, JSON, and InfluxDB output sinks."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from kasoku.benchmark import BENCHMARK_COLUMNS
from kasoku.config import InfluxDBSinkConfig, SinkConfig
from kasoku.forecast import FORECAST_COLUMNS
from kasoku.sinks import (
    CsvSink,
    InfluxDBSink,
    JsonSink,
    SinkError,
    create_sink,
    write_benchmark,
    write_forecast,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-05T00:00:00Z", "2024-01-05T01:00:00Z"]),
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
            "compile_overhead_seconds": [None, 0.5],
            "peak_memory_mb": [1.0, 2.0],
        }
    )


class _FakePoint:
    instances = []

    def __init__(self, measurement):
        self.measurement = measurement
        self.time_value = None
        self.tags = {}
        self.fields = {}
        _FakePoint.instances.append(self)

    def time(self, value):
        self.time_value = value
        return self

    def tag(self, key, value):
        self.tags[key] = value
        return self

    def field(self, key, value):
        self.fields[key] = value
        return self


class _FakeWriteApi:
    def __init__(self):
        self.writes = []

    def write(self, *, bucket, org, record):
        self.writes.append((bucket, org, record))


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        _FakeClient.instances.append(self)

    def write_api(self):
        return _FakeWriteApi.instance

    def close(self):
        self.closed = True


class _FakeWriteApi:
    instance = None

    def __init__(self):
        self.writes = []

    def write(self, *, bucket, org, record):
        self.writes.append((bucket, org, record))


def _install_fake_influxdb(monkeypatch):
    _FakePoint.instances = []
    _FakeClient.instances = []
    _FakeWriteApi.instance = _FakeWriteApi()
    monkeypatch.setitem(
        sys.modules,
        "influxdb_client",
        types.SimpleNamespace(InfluxDBClient=_FakeClient, Point=_FakePoint),
    )


def test_csv_sink_writes_forecast(tmp_path):
    path = tmp_path / "forecast.csv"
    CsvSink(path).write_forecast(_forecast_frame())
    read = pd.read_csv(path)
    assert list(read.columns) == FORECAST_COLUMNS
    assert read["symbol_or_series_id"].tolist() == ["series_single", "series_single"]
    assert read["horizon_step"].tolist() == [1, 2]
    assert read["predicted_value"].tolist() == [1.25, 1.5]
    assert read["model"].tolist() == ["gru", "gru"]
    assert (pd.to_datetime(read["timestamp"]) == _forecast_frame()["timestamp"]).all()


def test_csv_sink_writes_benchmark_with_nan_roundtrip(tmp_path):
    path = tmp_path / "benchmark.csv"
    CsvSink(path).write_benchmark(_benchmark_frame())
    read = pd.read_csv(path)
    assert list(read.columns) == BENCHMARK_COLUMNS
    assert pd.isna(read["compile_overhead_seconds"].iloc[0])
    assert read["compile_overhead_seconds"].iloc[1] == pytest.approx(0.5)
    assert read["wall_clock_seconds"].tolist() == pytest.approx([0.1, 0.2])
    assert read["peak_memory_mb"].tolist() == pytest.approx([1.0, 2.0])


def test_csv_sink_rejects_wrong_schema(tmp_path):
    with pytest.raises(SinkError, match="columns"):
        CsvSink(tmp_path / "out.csv").write_forecast(_benchmark_frame())
    with pytest.raises(SinkError, match="columns"):
        CsvSink(tmp_path / "out.csv").write_benchmark(_forecast_frame())


def test_json_sink_writes_forecast(tmp_path):
    path = tmp_path / "forecast.json"
    JsonSink(path).write_forecast(_forecast_frame())
    records = json.loads(path.read_text())
    assert len(records) == 2
    first = records[0]
    assert first["symbol_or_series_id"] == "series_single"
    assert first["horizon_step"] == 1
    assert first["predicted_value"] == 1.25
    assert first["model"] == "gru"
    assert pd.to_datetime(first["timestamp"]) == _forecast_frame()["timestamp"].iloc[0]


def test_json_sink_writes_benchmark_with_null(tmp_path):
    path = tmp_path / "benchmark.json"
    JsonSink(path).write_benchmark(_benchmark_frame())
    records = json.loads(path.read_text())
    assert len(records) == 2
    assert records[0]["compile_overhead_seconds"] is None
    assert records[1]["compile_overhead_seconds"] == 0.5
    assert records[1]["throughput_samples_per_sec"] == 80.0


def test_json_sink_rejects_wrong_schema(tmp_path):
    with pytest.raises(SinkError, match="columns"):
        JsonSink(tmp_path / "out.json").write_forecast(_benchmark_frame())


def test_influx_sink_missing_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "influxdb_client", None)
    sink = InfluxDBSink("http://localhost:8086", "t", "o", "b")
    with pytest.raises(SinkError, match="influxdb-client"):
        sink.write_forecast(_forecast_frame())
    with pytest.raises(SinkError, match="influxdb-client"):
        sink.write_benchmark(_benchmark_frame())


def test_influx_sink_writes_forecast_points(monkeypatch):
    _install_fake_influxdb(monkeypatch)
    sink = InfluxDBSink("http://localhost:8086", "t", "org1", "bucket1")
    sink.write_forecast(_forecast_frame())

    client = _FakeClient.instances[0]
    assert client.kwargs == {"url": "http://localhost:8086", "token": "t", "org": "org1"}
    assert client.closed
    bucket, org, points = _FakeWriteApi.instance.writes[0]
    assert bucket == "bucket1"
    assert org == "org1"
    assert [point.measurement for point in points] == ["kasoku_forecast"] * 2
    first = points[0]
    assert first.time_value == "2024-01-05T00:00:00+00:00"
    assert first.tags == {"symbol_or_series_id": "series_single", "model": "gru"}
    assert first.fields == {"horizon_step": 1, "predicted_value": 1.25}
    assert points[1].fields == {"horizon_step": 2, "predicted_value": 1.5}


def test_influx_sink_writes_benchmark_points_skipping_nan(monkeypatch):
    _install_fake_influxdb(monkeypatch)
    sink = InfluxDBSink("http://localhost:8086", "t", "o", "b")
    sink.write_benchmark(_benchmark_frame())

    _, _, points = _FakeWriteApi.instance.writes[0]
    assert [point.measurement for point in points] == ["kasoku_benchmark"] * 2
    eager, xla = points[0], points[1]
    assert eager.tags == {
        "run_id": "run-1",
        "model": "gru",
        "model_size": "small",
        "mode": "eager",
        "phase": "inference",
    }
    assert eager.fields == {
        "wall_clock_seconds": 0.1,
        "throughput_samples_per_sec": 160.0,
        "peak_memory_mb": 1.0,
    }
    assert "compile_overhead_seconds" not in eager.fields
    assert xla.fields["compile_overhead_seconds"] == 0.5
    assert eager.time_value == xla.time_value


def test_influx_sink_rejects_wrong_schema():
    sink = InfluxDBSink("http://localhost:8086", "t", "o", "b")
    with pytest.raises(SinkError, match="columns"):
        sink.write_forecast(_benchmark_frame())
    with pytest.raises(SinkError, match="columns"):
        sink.write_benchmark(_forecast_frame())


def test_create_sink_csv(tmp_path):
    output = tmp_path / "out.csv"
    sink = create_sink(SinkConfig(type="csv"), str(output))
    assert isinstance(sink, CsvSink)
    sink.write_forecast(_forecast_frame())
    assert output.exists()


def test_create_sink_json(tmp_path):
    output = tmp_path / "out.json"
    sink = create_sink(SinkConfig(type="json"), str(output))
    assert isinstance(sink, JsonSink)
    sink.write_benchmark(_benchmark_frame())
    assert output.exists()


def test_create_sink_influxdb():
    config = SinkConfig(
        type="influxdb",
        influxdb=InfluxDBSinkConfig(url="u", token="t", org="o", bucket="b"),
    )
    sink = create_sink(config)
    assert isinstance(sink, InfluxDBSink)
    assert sink.url == "u"
    assert sink.bucket == "b"


def test_create_sink_csv_requires_output():
    with pytest.raises(SinkError, match="output path"):
        create_sink(SinkConfig(type="csv"))
    with pytest.raises(SinkError, match="output path"):
        create_sink(SinkConfig(type="json"))


def test_create_sink_unknown_type():
    with pytest.raises(SinkError, match="unknown sink type"):
        create_sink(SinkConfig(type="parquet"))


def test_write_forecast_config_driven(tmp_path):
    output = tmp_path / "f.csv"
    write_forecast(_forecast_frame(), SinkConfig(type="csv"), str(output))
    read = pd.read_csv(output)
    assert list(read.columns) == FORECAST_COLUMNS
    assert read["predicted_value"].tolist() == [1.25, 1.5]


def test_write_benchmark_config_driven(tmp_path):
    output = tmp_path / "b.json"
    write_benchmark(_benchmark_frame(), SinkConfig(type="json"), str(output))
    records = json.loads(output.read_text())
    assert len(records) == 2
    assert records[1]["wall_clock_seconds"] == 0.2
