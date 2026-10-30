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


def test_main_slo_check_passing_frame(tmp_path, capsys):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n",
        encoding="utf-8",
    )
    assert main(["slo", "check", "--data", str(bench_csv)]) == 0
    out = capsys.readouterr().out
    assert "kasoku slo: PASS" in out


def test_main_slo_check_failing_frame(tmp_path, capsys):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.8,5.0,0.5,150.0\n",
        encoding="utf-8",
    )
    assert main(["slo", "check", "--data", str(bench_csv)]) == 1
    out = capsys.readouterr().out
    assert "kasoku slo: FAIL" in out


def test_main_slo_check_missing_data_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["slo", "check"])


def test_main_baseline_save_and_list(tmp_path):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n",
        encoding="utf-8",
    )
    baseline_json = tmp_path / "baselines.json"
    assert main(["baseline", "save", "--data", str(bench_csv), "--output", str(baseline_json)]) == 0
    assert baseline_json.exists()
    out = tmp_path / "list.out"  # not used; list writes to stdout
    rc = main(["baseline", "list", "--baseline", str(baseline_json)])
    assert rc == 0


def test_main_baseline_compare_passing_frame(tmp_path, capsys):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n",
        encoding="utf-8",
    )
    baseline_json = tmp_path / "baselines.json"
    assert main(["baseline", "save", "--data", str(bench_csv), "--output", str(baseline_json)]) == 0
    assert main(["baseline", "compare", "--data", str(bench_csv), "--baseline", str(baseline_json)]) == 0
    out = capsys.readouterr().out
    assert "kasoku baseline: PASS" in out


def test_main_baseline_compare_failing_frame(tmp_path, capsys):
    good_csv = tmp_path / "good.csv"
    good_csv.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n",
        encoding="utf-8",
    )
    regressed = tmp_path / "regressed.csv"
    regressed.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.8,5.0,0.5,150.0\n",
        encoding="utf-8",
    )
    baseline_json = tmp_path / "baselines.json"
    assert main(["baseline", "save", "--data", str(good_csv), "--output", str(baseline_json)]) == 0
    assert main(["baseline", "compare", "--data", str(regressed), "--baseline", str(baseline_json)]) == 1
    out = capsys.readouterr().out
    assert "kasoku baseline: FAIL" in out


def test_main_baseline_compare_missing_baseline_passes(tmp_path, capsys):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(
        "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
        "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
        "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.json"
    missing.write_text("{}\n", encoding="utf-8")
    assert main(["baseline", "compare", "--data", str(bench_csv), "--baseline", str(missing)]) == 0
    out = capsys.readouterr().out
    assert "kasoku baseline: PASS" in out


_ACCURACY_TRAINING = """\
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


def _accuracy_config(tmp_path, accuracy_section):
    path = tmp_path / "config.toml"
    path.write_text(f"{_ACCURACY_TRAINING}\n{accuracy_section}", encoding="utf-8")
    return str(path)


def test_main_accuracy_check_passing(tmp_path, capsys):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _accuracy_config(
        tmp_path,
        """
[accuracy]
mae = { enabled = true, threshold = 1000.0 }
mape = { enabled = false }
""",
    )
    output = tmp_path / "forecast.csv"
    assert (
        main(
            [
                "accuracy",
                "check",
                "--model",
                str(checkpoint),
                "--data",
                str(SINGLE),
                "--horizon",
                "24",
                "--config",
                config,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "kasoku accuracy: PASS" in out
    assert "mae: PASS" in out


def test_main_accuracy_check_failing(tmp_path, capsys):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _accuracy_config(
        tmp_path,
        """
[accuracy]
mae = { enabled = false }
mape = { enabled = true, threshold = 0.000001 }
""",
    )
    assert (
        main(
            [
                "accuracy",
                "check",
                "--model",
                str(checkpoint),
                "--data",
                str(SINGLE),
                "--horizon",
                "24",
                "--config",
                config,
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "kasoku accuracy: FAIL" in out


def test_main_accuracy_check_disabled_skipped(tmp_path, capsys):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _accuracy_config(
        tmp_path,
        """
[accuracy]
mae = { enabled = false }
mape = { enabled = false }
""",
    )
    assert (
        main(
            [
                "accuracy",
                "check",
                "--model",
                str(checkpoint),
                "--data",
                str(SINGLE),
                "--horizon",
                "24",
                "--config",
                config,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "kasoku accuracy: PASS" in out
    assert "mae: SKIP (disabled)" in out
    assert "mape: SKIP (disabled)" in out


def test_main_accuracy_check_missing_args_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["accuracy", "check"])


def test_main_accuracy_check_writes_output_json(tmp_path):
    checkpoint = _train_and_checkpoint(tmp_path)
    config = _accuracy_config(
        tmp_path,
        """
[accuracy]
mae = { enabled = true, threshold = 1000.0 }
mape = { enabled = false }
""",
    )
    output = tmp_path / "accuracy.json"
    assert (
        main(
            [
                "accuracy",
                "check",
                "--model",
                str(checkpoint),
                "--data",
                str(SINGLE),
                "--horizon",
                "24",
                "--config",
                config,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "mae" in data and "mape" in data
    assert "per_series" in data["mae"]


def test_main_accuracy_unknown_subcommand_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["accuracy", "bogus"])


_REPORT_CSV = (
    "run_id,model,model_size,mode,phase,wall_clock_seconds,throughput_samples_per_sec,compile_overhead_seconds,peak_memory_mb\n"
    "run-1,gru,small,eager,inference,2.0,10.0,,100.0\n"
    "run-1,gru,small,xla,inference,1.0,20.0,0.2,105.0\n"
)


def test_main_report_generate_writes_html(tmp_path, capsys):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(_REPORT_CSV, encoding="utf-8")
    output = tmp_path / "report.html"
    assert main(["report", "generate", "--data", str(bench_csv), "--output", str(output)]) == 0
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert 'id="chart-speedup"' in content
    assert 'id="chart-compile"' in content
    assert 'id="chart-throughput"' in content
    assert 'id="chart-memory"' in content
    assert "cdn.jsdelivr.net/npm/chart.js@4" in content
    out = capsys.readouterr().out
    assert "kasoku report: wrote HTML report" in out


def test_main_report_generate_with_compare(tmp_path):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(_REPORT_CSV, encoding="utf-8")
    compare_csv = tmp_path / "compare.csv"
    compare_csv.write_text(_REPORT_CSV, encoding="utf-8")
    output = tmp_path / "compare.html"
    assert main(["report", "generate", "--data", str(bench_csv), "--compare", str(compare_csv), "--output", str(output)]) == 0
    assert output.exists()


def test_main_report_generate_with_accuracy(tmp_path):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(_REPORT_CSV, encoding="utf-8")
    accuracy_json = tmp_path / "accuracy.json"
    accuracy_json.write_text(
        json.dumps(
            {
                "passed": True,
                "mae": {
                    "value": 1.0,
                    "threshold": 2.0,
                    "passed": True,
                    "enabled": True,
                    "per_series": {"s1": 1.0},
                },
                "mape": {
                    "value": 0.05,
                    "threshold": 0.15,
                    "passed": True,
                    "enabled": True,
                    "per_series": {"s1": 0.05},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    assert (
        main(
            [
                "report",
                "generate",
                "--data",
                str(bench_csv),
                "--accuracy",
                str(accuracy_json),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert 'id="chart-mae"' in content
    assert 'id="chart-mape"' in content
    assert 'id="chart-accuracy-summary"' in content


def test_main_report_generate_with_compare_accuracy(tmp_path):
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text(_REPORT_CSV, encoding="utf-8")
    accuracy_json = tmp_path / "accuracy.json"
    accuracy_json.write_text(
        json.dumps(
            {
                "passed": True,
                "mae": {
                    "value": 1.0,
                    "threshold": 2.0,
                    "passed": True,
                    "enabled": True,
                    "per_series": {"s1": 1.0},
                },
                "mape": {
                    "value": 0.05,
                    "threshold": 0.15,
                    "passed": True,
                    "enabled": True,
                    "per_series": {"s1": 0.05},
                },
            }
        ),
        encoding="utf-8",
    )
    compare_accuracy_json = tmp_path / "compare_accuracy.json"
    compare_accuracy_json.write_text(
        json.dumps(
            {
                "passed": True,
                "mae": {
                    "value": 1.5,
                    "threshold": 2.0,
                    "passed": True,
                    "enabled": True,
                    "per_series": {"s1": 1.5},
                },
                "mape": {
                    "value": 0.1,
                    "threshold": 0.15,
                    "passed": True,
                    "enabled": True,
                    "per_series": {"s1": 0.1},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "compare.html"
    assert (
        main(
            [
                "report",
                "generate",
                "--data",
                str(bench_csv),
                "--accuracy",
                str(accuracy_json),
                "--compare-accuracy",
                str(compare_accuracy_json),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "const COMPARE_ACCURACY_DATA = " in content


def test_main_report_generate_missing_data_returns_one(tmp_path, capsys):
    output = tmp_path / "report.html"
    rc = main(["report", "generate", "--data", str(tmp_path / "missing.csv"), "--output", str(output)])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_main_report_unknown_subcommand_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["report", "bogus"])


