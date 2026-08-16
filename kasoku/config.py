"""TOML configuration schema and loader for Kasoku.

The config file is optional -- every value has a sensible default -- and
covers three sections:

- ``training``: model, execution mode, training hyperparameters
- ``benchmark``: sweep dimensions (model sizes, modes, phases) and repetition
- ``sink``: where forecast and benchmark output is written

Sink output is CSV/JSON by default; InfluxDB is an optional, config-driven
sink and never a hard dependency.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_MODELS = frozenset({"gru", "lstm", "tcn"})
VALID_MODES = frozenset({"eager", "tf_function", "xla"})
VALID_SINKS = frozenset({"csv", "json", "influxdb"})
VALID_PHASES = frozenset({"train", "inference"})


class ConfigError(ValueError):
    """Raised when configuration is invalid or cannot be loaded."""


@dataclass
class TrainingConfig:
    model: str = "gru"
    mode: str = "eager"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    seed: int = 42
    window_size: int = 24
    horizon: int = 1
    checkpoint_dir: str = "checkpoints"


@dataclass
class BenchmarkConfig:
    model: str = "gru"
    sizes: list[str] = field(default_factory=lambda: ["small", "medium"])
    modes: list[str] = field(default_factory=lambda: ["eager", "tf_function", "xla"])
    phases: list[str] = field(default_factory=lambda: ["train", "inference"])
    iterations: int = 3
    warmup_iterations: int = 1


@dataclass
class InfluxDBSinkConfig:
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = ""
    bucket: str = "kasoku"


@dataclass
class SinkConfig:
    type: str = "csv"
    influxdb: InfluxDBSinkConfig = field(default_factory=InfluxDBSinkConfig)


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a TOML config file into a :class:`Config`."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    try:
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {cfg_path}: {exc}") from exc
    return parse_config(data)


def parse_config(data: Mapping[str, Any]) -> Config:
    """Parse and validate a raw config mapping into a :class:`Config`."""
    if not isinstance(data, Mapping):
        raise ConfigError("config root must be a TOML table")
    return Config(
        training=_parse_training(data.get("training", {})),
        benchmark=_parse_benchmark(data.get("benchmark", {})),
        sink=_parse_sink(data.get("sink", {})),
    )


def _parse_training(raw: Any) -> TrainingConfig:
    section = _as_table(raw, "training")
    model = _typed_str(section, "model", TrainingConfig().model)
    mode = _typed_str(section, "mode", TrainingConfig().mode)
    if model not in VALID_MODELS:
        raise ConfigError(f"training.model must be one of {sorted(VALID_MODELS)}, got {model!r}")
    if mode not in VALID_MODES:
        raise ConfigError(f"training.mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    return TrainingConfig(
        model=model,
        mode=mode,
        epochs=_typed_int(section, "epochs", default=50, minimum=1),
        batch_size=_typed_int(section, "batch_size", default=32, minimum=1),
        learning_rate=_typed_float(section, "learning_rate", default=1e-3, minimum=0.0),
        early_stopping_patience=_typed_int(
            section, "early_stopping_patience", default=5, minimum=0
        ),
        seed=_typed_int(section, "seed", default=42),
        window_size=_typed_int(section, "window_size", default=24, minimum=1),
        horizon=_typed_int(section, "horizon", default=1, minimum=1),
        checkpoint_dir=_typed_str(section, "checkpoint_dir", default="checkpoints"),
    )


def _parse_benchmark(raw: Any) -> BenchmarkConfig:
    section = _as_table(raw, "benchmark")
    model = _typed_str(section, "model", "gru")
    if model not in VALID_MODELS:
        raise ConfigError(f"benchmark.model must be one of {sorted(VALID_MODELS)}, got {model!r}")
    modes = _typed_str_list(section, "modes", default=["eager", "tf_function", "xla"])
    invalid_modes = sorted(set(modes) - VALID_MODES)
    if invalid_modes:
        raise ConfigError(
            f"benchmark.modes contains invalid values: {invalid_modes} "
            f"(allowed: {sorted(VALID_MODES)})"
        )
    phases = _typed_str_list(section, "phases", default=["train", "inference"])
    invalid_phases = sorted(set(phases) - VALID_PHASES)
    if invalid_phases:
        raise ConfigError(
            f"benchmark.phases contains invalid values: {invalid_phases} "
            f"(allowed: {sorted(VALID_PHASES)})"
        )
    return BenchmarkConfig(
        model=model,
        sizes=_typed_str_list(section, "sizes", default=["small", "medium"]),
        modes=modes,
        phases=phases,
        iterations=_typed_int(section, "iterations", default=3, minimum=1),
        warmup_iterations=_typed_int(section, "warmup_iterations", default=1, minimum=0),
    )


def _parse_sink(raw: Any) -> SinkConfig:
    section = _as_table(raw, "sink")
    sink_type = _typed_str(section, "type", "csv")
    if sink_type not in VALID_SINKS:
        raise ConfigError(f"sink.type must be one of {sorted(VALID_SINKS)}, got {sink_type!r}")
    influx_raw = section.get("influxdb", {})
    influx = InfluxDBSinkConfig(
        url=_typed_str(influx_raw, "url", "http://localhost:8086"),
        token=_typed_str(influx_raw, "token", ""),
        org=_typed_str(influx_raw, "org", ""),
        bucket=_typed_str(influx_raw, "bucket", "kasoku"),
    )
    return SinkConfig(type=sink_type, influxdb=influx)


def _as_table(raw: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{section} must be a TOML table, got {type(raw).__name__}")
    return raw


def _typed_str(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _typed_str_list(section: Mapping[str, Any], key: str, default: list[str]) -> list[str]:
    value = section.get(key, default)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{key} must be an array of strings")
    return list(value)


def _typed_int(section: Mapping[str, Any], key: str, default: int, minimum: int | None = None) -> int:
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def _typed_float(section: Mapping[str, Any], key: str, default: float, minimum: float | None = None) -> float:
    value = section.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be a number, got {type(value).__name__}")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value
