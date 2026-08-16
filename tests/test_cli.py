"""End-to-end tests running train, forecast, and benchmark from fixture data."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from kasoku.benchmark import BENCHMARK_COLUMNS
from kasoku.cli import main
from kasoku.data import CsvSource
from kasoku.forecast import FORECAST_COLUMNS

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"

TRAINING_TOML = """\
[training]
model = "gru"
size = "small"
mode = "eager"
epochs = 2
batch_size = 16
seed = 42
window_size = 8
horizon = 1
"""

BENCHMARK_TOML = """\
[benchmark]
model = "gru"
sizes = ["small"]
modes = ["eager"]
phases = ["inference"]
iterations = 1
warmup_iterations = 0
"""


def _write_config(tmp_path, training="", benchmark="", sink=""):
    path = tmp_path / "config.toml"
    path.write_text(f"{training}\n{benchmark}\n{sink}", encoding="utf-8")
    return str(path)


def _train_kwargs(tmp_path, config=None, output="model.weights.h5", extra=None):
    return [
        "train",
        "--data", str(SINGLE),
        "--output", str(tmp_path / output),
        *(["--config", config] if config else []),
        *(extra or []),
    ]


def _train_and_checkpoint(tmp_path):
    config = _write_config(tmp_path, training=TRAINING_TOML)
    checkpoint = tmp_path / "model.weights.h5"
    assert main(_train_kwargs(tmp_path, config=config, output=str(checkpoint))) == 0
    return checkpoint


class _FakePoint:
    def __init__(self, measurement):
        self.measurement = measurement
        self.tags = {}
        self.fields = {}

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
    instance = None

    def __init__(self):
        self.writes = []

    def write(self, *, bucket, org, record):
        self.writes.append((bucket, org, record))


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeClient.instances.append(self)

    def write_api(self):
        return _FakeWriteApi.instance

    def close(self):
        pass


def _install_fake_influxdb(monkeypatch):
    _FakeWriteApi.instance = _FakeWriteApi()
    _FakeClient.instances = []
    monkeypatch.setitem(
        sys.modules,
        "influxdb_client",
        types.SimpleNamespace(InfluxDBClient=_FakeClient, Point=_FakePoint),
    )


def _influxdb_config(tmp_path):
    return _write_config(
        tmp_path,
        training=TRAINING_TOML,
        benchmark=BENCHMARK_TOML,
        sink='[sink]\ntype = "influxdb"\n\n[sink.influxdb]\norg = "org1"\nbucket = "bucket1"\n',
    )


def test_main_train_saves_checkpoint_and_returns_zero(tmp_path, capsys):
    config = _write_config(tmp_path, training=TRAINING_TOML)
    checkpoint = tmp_path / "model.weights.h5"
    assert main(_train_kwargs(tmp_path, config=config, output=str(checkpoint), extra=["--verbose"])) == 0
    assert checkpoint.exists()
    assert "epoch 1/2" in capsys.readouterr().out


def test_main_train_without_config_uses_defaults(tmp_path):
    assert main(_train_kwargs(tmp_path, output=str(tmp_path / "no_config.weights.h5"))) == 0
    assert (tmp_path / "no_config.weights.h5").exists()


def test_main_train_unknown_model_returns_one(tmp_path, capsys):
    config = _write_config(tmp_path, training=TRAINING_TOML)
    rc = main(["train", "--data", str(SINGLE), "--model", "transformer", "--config", config, "--output", str(tmp_path / "x.weights.h5")])
    assert rc == 1
    assert "unknown model" in capsys.readouterr().err


def test_main_train_invalid_mode_returns_one(tmp_path, capsys):
    config = _write_config(tmp_path, training=TRAINING_TOML)
    rc = main(_train_kwargs(tmp_path, config=config, extra=["--mode", "quantum"]))
    assert rc == 1
    assert "unknown execution mode" in capsys.readouterr().err


def test_main_train_missing_config_returns_one(tmp_path, capsys):
    rc = main(["train", "--data", str(SINGLE), "--config", str(tmp_path / "nope.toml"), "--output", str(tmp_path / "x.weights.h5")])
    assert rc == 1
    assert "config file not found" in capsys.readouterr().err


def test_main_forecast_writes_csv_end_to_end(tmp_path):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _write_config(tmp_path, training=TRAINING_TOML)
    output = tmp_path / "forecast.csv"
    rc = main(["forecast", "--model", str(checkpoint), "--data", str(SINGLE), "--horizon", "3", "--config", config, "--output", str(output)])
    assert rc == 0
    read = pd.read_csv(output)
    assert list(read.columns) == FORECAST_COLUMNS
    assert len(read) == 3
    assert read["horizon_step"].tolist() == [1, 2, 3]


def test_main_forecast_writes_json(tmp_path):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _write_config(tmp_path, training=TRAINING_TOML)
    output = tmp_path / "forecast.json"
    assert main(["forecast", "--model", str(checkpoint), "--data", str(SINGLE), "--horizon", "2", "--config", config, "--output", str(output)]) == 0
    records = json.loads(output.read_text())
    assert len(records) == 2
    assert records[0]["predicted_value"] is not None


def test_main_forecast_missing_checkpoint_returns_one(tmp_path, capsys):
    config = _write_config(tmp_path, training=TRAINING_TOML)
    rc = main(["forecast", "--model", str(tmp_path / "nope.weights.h5"), "--data", str(SINGLE), "--horizon", "2", "--config", config, "--output", str(tmp_path / "f.csv")])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_main_forecast_invalid_output_returns_one(tmp_path, capsys):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _write_config(tmp_path, training=TRAINING_TOML)
    rc = main(["forecast", "--model", str(checkpoint), "--data", str(SINGLE), "--horizon", "2", "--config", config, "--output", str(tmp_path / "f.txt")])
    assert rc == 1
    assert "cannot infer output sink" in capsys.readouterr().err


def test_main_forecast_influxdb_sink(tmp_path, monkeypatch):
    _install_fake_influxdb(monkeypatch)
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _influxdb_config(tmp_path)
    output = "http://localhost:9999"
    assert main(["forecast", "--model", str(checkpoint), "--data", str(SINGLE), "--horizon", "2", "--config", config, "--output", output]) == 0
    bucket, org, points = _FakeWriteApi.instance.writes[0]
    assert bucket == "bucket1"
    assert org == "org1"
    assert [point.measurement for point in points] == ["kasoku_forecast"] * 2


def test_main_benchmark_writes_csv(tmp_path):
    config = _write_config(tmp_path, training=TRAINING_TOML, benchmark=BENCHMARK_TOML)
    output = tmp_path / "benchmark.csv"
    rc = main(["benchmark", "--data", str(SINGLE), "--config", config, "--output", str(output)])
    assert rc == 0
    read = pd.read_csv(output)
    assert list(read.columns) == BENCHMARK_COLUMNS
    assert len(read) == 1
    assert read["mode"].iloc[0] == "eager"


def test_main_benchmark_flags_without_config(tmp_path):
    output = tmp_path / "benchmark.csv"
    rc = main(["benchmark", "--data", str(SINGLE), "--sizes", "small", "--modes", "eager", "--phases", "inference", "--output", str(output)])
    assert rc == 0
    read = pd.read_csv(output)
    assert len(read) == 1
    assert read["model_size"].iloc[0] == "small"


def test_main_benchmark_flags_override_config(tmp_path):
    config = _write_config(tmp_path, training=TRAINING_TOML, benchmark=BENCHMARK_TOML)
    output = tmp_path / "benchmark.csv"
    rc = main(["benchmark", "--data", str(SINGLE), "--config", config, "--sizes", "medium", "--output", str(output)])
    assert rc == 0
    read = pd.read_csv(output)
    assert len(read) == 1
    assert read["model_size"].iloc[0] == "medium"


def test_main_benchmark_influxdb_sink(tmp_path, monkeypatch):
    _install_fake_influxdb(monkeypatch)
    config = _influxdb_config(tmp_path)
    assert main(["benchmark", "--data", str(SINGLE), "--config", config, "--output", "http://localhost:9999"]) == 0
    bucket, org, points = _FakeWriteApi.instance.writes[0]
    assert bucket == "bucket1"
    assert org == "org1"
    assert [point.measurement for point in points] == ["kasoku_benchmark"]


def test_main_benchmark_unknown_model_returns_one(tmp_path, capsys):
    config = _write_config(tmp_path, training=TRAINING_TOML, benchmark=BENCHMARK_TOML)
    rc = main(["benchmark", "--data", str(SINGLE), "--config", config, "--model", "rnn", "--output", str(tmp_path / "b.csv")])
    assert rc == 1
    assert "unknown model" in capsys.readouterr().err


def test_main_no_args_raises_system_exit():
    with pytest.raises(SystemExit):
        main([])


def test_main_help_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["--help"])


def test_main_train_missing_required_args_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["train"])


def test_cli_reads_fixture_series():
    df = CsvSource(SINGLE).read()
    assert len(df) == 96
