"""Smoke tests for the TOML config schema and loader."""

from __future__ import annotations

import pytest

from kasoku.config import (
    ConfigError,
    load_config,
    parse_config,
)


def test_load_full_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[training]
model = "lstm"
size = "medium"
mode = "xla"
epochs = 100
batch_size = 16
learning_rate = 0.0005
early_stopping_patience = 10
seed = 7
window_size = 48
horizon = 12
checkpoint_dir = "artifacts"

[benchmark]
model = "lstm"
sizes = ["small"]
modes = ["eager", "xla"]
phases = ["inference"]
iterations = 5
warmup_iterations = 2

[sink]
type = "influxdb"

[sink.influxdb]
url = "http://localhost:9999"
token = "secret"
org = "acme"
bucket = "telemetry"
""",
        encoding="utf-8",
    )
    config = load_config(cfg)

    assert config.training.model == "lstm"
    assert config.training.size == "medium"
    assert config.training.mode == "xla"
    assert config.training.epochs == 100
    assert config.training.batch_size == 16
    assert config.training.learning_rate == 0.0005
    assert config.training.early_stopping_patience == 10
    assert config.training.seed == 7
    assert config.training.window_size == 48
    assert config.training.horizon == 12
    assert config.training.checkpoint_dir == "artifacts"

    assert config.benchmark.model == "lstm"
    assert config.benchmark.sizes == ["small"]
    assert config.benchmark.modes == ["eager", "xla"]
    assert config.benchmark.phases == ["inference"]
    assert config.benchmark.iterations == 5
    assert config.benchmark.warmup_iterations == 2

    assert config.sink.type == "influxdb"
    assert config.sink.influxdb.url == "http://localhost:9999"
    assert config.sink.influxdb.token == "secret"
    assert config.sink.influxdb.org == "acme"
    assert config.sink.influxdb.bucket == "telemetry"


def test_empty_config_uses_defaults(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("", encoding="utf-8")
    config = load_config(cfg)

    assert config.training.model == "gru"
    assert config.training.size == "small"
    assert config.training.mode == "eager"
    assert config.training.epochs == 50
    assert config.benchmark.sizes == ["small", "medium"]
    assert config.benchmark.modes == ["eager", "tf_function", "xla"]
    assert config.sink.type == "csv"
    assert config.sink.influxdb.url == "http://localhost:8086"


def test_partial_sections_merge_with_defaults(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[training]\nmodel = "tcn"\n', encoding="utf-8")
    config = load_config(cfg)

    assert config.training.model == "tcn"
    assert config.training.mode == "eager"
    assert config.benchmark.model == "gru"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_raises(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("this is { not ] valid toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg)


@pytest.mark.parametrize("model", ["transformer", "grux", ""])
def test_invalid_model_raises(model):
    with pytest.raises(ConfigError):
        parse_config({"training": {"model": model}})


@pytest.mark.parametrize("model", ["transformer", "grux", ""])
def test_invalid_benchmark_model_raises(model):
    with pytest.raises(ConfigError):
        parse_config({"benchmark": {"model": model}})


@pytest.mark.parametrize("mode", ["graph", "jit", ""])
def test_invalid_mode_raises(mode):
    with pytest.raises(ConfigError):
        parse_config({"training": {"mode": mode}})


@pytest.mark.parametrize("sink_type", ["sqlite", "parquet", ""])
def test_invalid_sink_type_raises(sink_type):
    with pytest.raises(ConfigError):
        parse_config({"sink": {"type": sink_type}})


def test_invalid_benchmark_mode_raises():
    with pytest.raises(ConfigError):
        parse_config({"benchmark": {"modes": ["eager", "cuda"]}})


def test_invalid_benchmark_phase_raises():
    with pytest.raises(ConfigError):
        parse_config({"benchmark": {"phases": ["train", "validate"]}})


def test_non_table_section_raises():
    with pytest.raises(ConfigError):
        parse_config({"training": "fast"})


def test_non_mapping_root_raises():
    with pytest.raises(ConfigError):
        parse_config(["training"])


def test_wrong_type_raises():
    with pytest.raises(ConfigError):
        parse_config({"training": {"model": 42}})
    with pytest.raises(ConfigError):
        parse_config({"training": {"epochs": "many"}})
    with pytest.raises(ConfigError):
        parse_config({"benchmark": {"sizes": ["small", 3]}})
    with pytest.raises(ConfigError):
        parse_config({"training": {"learning_rate": "fast"}})


@pytest.mark.parametrize(
    "field",
    ["epochs", "batch_size", "window_size", "horizon"],
)
def test_non_positive_int_raises(field):
    with pytest.raises(ConfigError):
        parse_config({"training": {field: 0}})


def test_negative_learning_rate_raises():
    with pytest.raises(ConfigError):
        parse_config({"training": {"learning_rate": -0.1}})


def test_bool_not_accepted_as_int():
    with pytest.raises(ConfigError):
        parse_config({"training": {"epochs": True}})
