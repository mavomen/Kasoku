"""Command-line interface for Kasoku.

Three subcommands -- ``train``, ``forecast``, and ``benchmark`` -- expose the
pipeline, models, execution modes, and sinks from the command line. ``main``
returns an exit code (0 on success, 1 on any known error), so it works as a
console-script entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from kasoku.alerts import AlertsError, render_alerts_yaml
from kasoku.benchmark import BenchmarkError, run_benchmark
from kasoku.config import AlertsConfig, Config, ConfigError, load_config
from kasoku.data import CsvSource, DataError
from kasoku.execution import ExecutionError
from kasoku.forecast import ForecastError, forecast_series, load_trained_model
from kasoku.models import ModelError, build_gru_by_size, build_lstm_by_size, build_tcn_by_size
from kasoku.observability import PushgatewayError, push_benchmark_metrics
from kasoku.pipeline import PipelineError, window_series
from kasoku.sinks import CsvSink, InfluxDBSink, JsonSink, SinkError
from kasoku.slo import SLOError, all_passed, evaluate_slo
from kasoku.train import TrainError, train

_KNOWN_ERRORS = (
    ConfigError,
    DataError,
    ExecutionError,
    ForecastError,
    ModelError,
    PipelineError,
    TrainError,
    BenchmarkError,
    SinkError,
    PushgatewayError,
    AlertsError,
    SLOError,
    OSError,
)

_MODEL_BUILDERS = {
    "gru": build_gru_by_size,
    "lstm": build_lstm_by_size,
    "tcn": build_tcn_by_size,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the requested subcommand, returning an exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return _train(args)
        if args.command == "forecast":
            return _forecast(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "alerts":
            if args.alerts_command == "render":
                return _alerts_render(args)
            if args.alerts_command == "validate":
                return _alerts_validate(args)
            return _alerts_show(args)
        if args.command == "slo":
            if args.slo_command == "check":
                return _slo_check(args)
    except _KNOWN_ERRORS as exc:
        print(f"kasoku: error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the ``kasoku`` argument parser with its three subcommands."""
    parser = argparse.ArgumentParser(
        prog="kasoku",
        description="Time-Series Forecasting & XLA Benchmark Platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train a forecasting model")
    train.add_argument("--data", required=True, help="input series csv path")
    train.add_argument("--model", default=None, help="model name (gru|lstm|tcn); default from config")
    train.add_argument("--mode", default=None, help="execution mode (eager|tf_function|xla); default from config")
    train.add_argument("--config", default=None, help="path to config.toml")
    train.add_argument("--output", required=True, help="checkpoint destination path")
    train.add_argument("--verbose", action="store_true", help="print per-epoch training loss")

    forecast = subparsers.add_parser("forecast", help="forecast from a trained checkpoint")
    forecast.add_argument("--model", required=True, help="trained checkpoint path")
    forecast.add_argument("--data", required=True, help="input series csv path")
    forecast.add_argument("--horizon", type=int, required=True, help="forecast steps ahead")
    forecast.add_argument("--mode", default=None, help="execution mode (eager|tf_function|xla); default from config")
    forecast.add_argument("--config", default=None, help="path to config.toml")
    forecast.add_argument("--output", required=True, help="forecast output path (.csv/.json) or influxdb url")

    benchmark = subparsers.add_parser("benchmark", help="run a model x size x mode x phase benchmark sweep")
    benchmark.add_argument("--data", required=True, help="input series csv path")
    benchmark.add_argument("--model", default=None, help="model name (gru|lstm|tcn); default from config")
    benchmark.add_argument("--sizes", default=None, help="comma-separated model sizes (default from config)")
    benchmark.add_argument("--modes", default=None, help="comma-separated execution modes (default from config)")
    benchmark.add_argument("--phases", default=None, help="comma-separated phases (train,inference) (default from config)")
    benchmark.add_argument("--config", default=None, help="path to config.toml")
    benchmark.add_argument("--output", required=True, help="benchmark output path (.csv/.json) or influxdb url")
    benchmark.add_argument(
        "--push-gateway",
        default=None,
        help="Pushgateway base URL (e.g. http://localhost:9091); pushes benchmark metrics after writing output",
    )

    alerts = subparsers.add_parser("alerts", help="render and validate Prometheus alert rules")
    alerts_sub = alerts.add_subparsers(dest="alerts_command", required=True)

    alerts_render = alerts_sub.add_parser("render", help="render alert rules YAML from config")
    alerts_render.add_argument("--config", default=None, help="path to config.toml")
    alerts_render.add_argument(
        "--output",
        default=None,
        help="write rendered YAML to this file instead of stdout",
    )

    alerts_validate = alerts_sub.add_parser("validate", help="validate the [alerts] config section")
    alerts_validate.add_argument("--config", default=None, help="path to config.toml")

    alerts_show = alerts_sub.add_parser("show", help="print resolved alert thresholds")
    alerts_show.add_argument("--config", default=None, help="path to config.toml")

    slo = subparsers.add_parser("slo", help="evaluate service level objectives against benchmark results")
    slo_sub = slo.add_subparsers(dest="slo_command", required=True)

    slo_check = slo_sub.add_parser("check", help="evaluate SLO targets against a benchmark output CSV")
    slo_check.add_argument("--config", default=None, help="path to config.toml")
    slo_check.add_argument(
        "--data",
        required=True,
        help="benchmark results CSV path to evaluate against SLO targets",
    )

    return parser


def _load_config(args) -> Config:
    """Load the configured TOML file, or return built-in defaults."""
    if args.config is None:
        return Config()
    return load_config(args.config)


def _split_csv(value: str | None, default: Sequence[str]) -> Sequence[str]:
    """Split a comma-separated CLI list, falling back to the config default."""
    if value is None:
        return default
    return [item.strip() for item in value.split(",")]


def _train(args) -> int:
    config = _load_config(args)
    model_name = args.model or config.training.model
    mode = args.mode or config.training.mode
    lookback = config.training.window_size
    horizon = config.training.horizon
    df = CsvSource(args.data).read()
    x, y = window_series(df, lookback=lookback, horizon=horizon)
    model = _build_model(model_name, config.training.size, lookback, horizon, config.training.seed)
    result = train(
        model,
        x,
        y,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        mode=mode,
        seed=config.training.seed,
        early_stopping_patience=config.training.early_stopping_patience,
        checkpoint_dir=None,
        verbose=args.verbose,
    )
    model.save_weights(args.output)
    detail = f" best_val_loss={result.best_val_loss:.6f}" if result.best_val_loss is not None else ""
    print(
        f"trained {model_name}/{config.training.size}: {result.epochs_trained} epochs"
        f"{detail}; checkpoint -> {args.output}"
    )
    return 0


def _build_model(model_name: str, size: str, lookback: int, horizon: int, seed: int | None):
    try:
        builder = _MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ModelError(
            f"unknown model {model_name!r}; expected one of {sorted(_MODEL_BUILDERS)}"
        ) from exc
    return builder(size, lookback=lookback, horizon=horizon, seed=seed)


def _forecast(args) -> int:
    config = _load_config(args)
    model = load_trained_model(
        config.training.model,
        config.training.size,
        lookback=config.training.window_size,
        horizon=config.training.horizon,
        checkpoint_path=args.model,
        seed=config.training.seed,
    )
    df = CsvSource(args.data).read()
    frame = forecast_series(model, df, steps=args.horizon, mode=args.mode or config.training.mode)
    _write_output(frame, config, args.output, kind="forecast")
    print(f"wrote {len(frame)} forecast rows to {args.output}")
    return 0


def _write_output(frame, config: Config, output: str, kind: str) -> None:
    """Route an output frame through the sink inferred from config/output."""
    if config.sink.type == "influxdb":
        influx = config.sink.influxdb
        sink = InfluxDBSink(
            url=output,
            token=influx.token,
            org=influx.org,
            bucket=influx.bucket,
        )
    elif output.endswith(".csv"):
        sink = CsvSink(output)
    elif output.endswith(".json"):
        sink = JsonSink(output)
    else:
        raise SinkError(
            f"cannot infer output sink from {output!r}; use a .csv/.json path "
            'or configure sink.type = "influxdb"'
        )
    if kind == "forecast":
        sink.write_forecast(frame)
    else:
        sink.write_benchmark(frame)


def _benchmark(args) -> int:
    config = _load_config(args)
    df = CsvSource(args.data).read()
    model_name = args.model or config.benchmark.model
    frame = run_benchmark(
        df,
        model_name,
        sizes=_split_csv(args.sizes, config.benchmark.sizes),
        modes=_split_csv(args.modes, config.benchmark.modes),
        phases=_split_csv(args.phases, config.benchmark.phases),
        lookback=config.training.window_size,
        horizon=config.training.horizon,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
        iterations=config.benchmark.iterations,
        warmup_iterations=config.benchmark.warmup_iterations,
    )
    _write_output(frame, config, args.output, kind="benchmark")
    if args.push_gateway is not None:
        push_benchmark_metrics(frame, args.push_gateway)
    print(
        f"wrote {len(frame)} benchmark rows ({model_name} x "
        f"{sorted(set(frame['model_size']))}) to {args.output}"
    )
    return 0


def _load_alerts_config(args) -> AlertsConfig:
    """Load the resolved :class:`AlertsConfig` from config or built-in defaults."""
    if args.config is None:
        return Config().alerts
    return load_config(args.config).alerts


def _alerts_render(args) -> int:
    config = _load_alerts_config(args)
    yaml_text = render_alerts_yaml(config)
    if args.output is not None:
        Path(args.output).write_text(yaml_text, encoding="utf-8")
    else:
        print(yaml_text, end="")
    return 0


def _alerts_validate(args) -> int:
    _load_alerts_config(args)
    print("kasoku alerts: configuration is valid")
    return 0


def _alerts_show(args) -> int:
    config = _load_alerts_config(args)
    rules = (
        ("xla_no_speedup", config.xla_no_speedup),
        ("compile_overhead", config.compile_overhead),
        ("xla_regression", config.xla_regression),
        ("throughput_regression", config.throughput_regression),
    )
    for name, rule in rules:
        parts = [f"enabled={rule.enabled}"]
        if rule.ratio is not None:
            parts.append(f"ratio={rule.ratio}")
        if rule.multiplier is not None:
            parts.append(f"multiplier={rule.multiplier}")
        print(f"{name}: {', '.join(parts)}")
    return 0


def _slo_check(args) -> int:
    config = _load_config(args)
    df = pd.read_csv(args.data)
    results = evaluate_slo(df, config.slo)
    all_ok = all_passed(results)
    status = "PASS" if all_ok else "FAIL"
    print(f"kasoku slo: {status}")
    for name, entry in results.items():
        value_str = "N/A" if entry["value"] is None else f"{entry['value']:.4f}"
        passed_str = "PASS" if entry["passed"] else "FAIL"
        print(
            f"  {name}: {passed_str} "
            f"(value={value_str}, target={entry['target']}, op={entry['operator']})"
        )
        for combo, combo_result in entry["combos"].items():
            combo_value = "N/A" if combo_result["value"] is None else f"{combo_result['value']:.4f}"
            combo_passed = "PASS" if combo_result["passed"] else "FAIL"
            print(f"    {combo}: {combo_passed} (value={combo_value})")
    return 0 if all_ok else 1
