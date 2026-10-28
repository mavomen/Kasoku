# Kasoku

Time-Series Forecasting & XLA Benchmark Platform

**Kasoku** (加速) means "acceleration" — both the compute speedup from XLA compilation and the act of forecasting ahead in time.

Kasoku is a standalone time-series forecasting platform built on TensorFlow. Model performance isn't just accuracy — it's also how fast you can train and serve that model, and how much of that speed is left on the table by not compiling it. Kasoku ships a small set of forecasting architectures runnable in three execution modes — eager, `tf.function`, and `tf.function(jit_compile=True)` (XLA) — with a first-class benchmark suite that measures and reports the difference.

It's input-agnostic (CSV, InfluxDB, or any Pandas-compatible source) and output-flexible (CSV/JSON by default, InfluxDB optional), CLI-first, and designed to stand on its own.

## Execution modes

Every model in `kasoku/models/` is implemented once and runnable in all three modes via the wrappers in `kasoku/execution/`. Mode is a runtime/config choice, not a separate code path:

| Mode | Description |
|------|-------------|
| `eager` | Baseline, easiest to debug, slowest |
| `tf_function` | Graph-compiled, no XLA |
| `xla` | `tf.function(jit_compile=True)` — XLA-compiled, expected fastest steady-state, with first-call compilation overhead reported separately |

The same model produces numerically equivalent outputs (within floating-point tolerance) across eager, `tf.function`, and XLA — proving modes are a runtime choice, not three separate implementations.

## Repository structure

```
kasoku/
├── data/              # data sources: csv, influxdb, generic pandas adapter
├── pipeline/           # windowing, train/val/test split, normalization
├── models/             # GRU, LSTM, temporal conv — single impl per model
├── execution/          # eager / tf.function / xla mode wrappers
├── benchmark/          # benchmark suite: time, throughput, compile overhead, memory
├── forecast/           # inference / forecasting logic
├── sinks/              # output writers: csv, json, influxdb
├── train/              # model training
└── cli/                # cli entrypoint
```

## Usage

```
kasoku train --data series.csv --model gru --mode xla \
  --config config.toml --output model.ckpt

kasoku forecast --model model.ckpt --data series.csv \
  --horizon 24 --output forecast.csv

kasoku benchmark --model gru --sizes small,medium \
  --modes eager,tf_function,xla --phases train,inference \
  --output benchmarks/gru_results.csv
```

`--output` accepts a path (CSV/JSON inferred from extension) or an InfluxDB connection string when the optional sink is configured in `config.toml`.

## Output schema

**Forecast output (CSV/JSON)**

| field | type | notes |
|-------|------|-------|
| `timestamp` | datetime | forecasted timestamp |
| `symbol_or_series_id` | string | identifier for the series |
| `horizon_step` | int | steps ahead from forecast origin |
| `predicted_value` | float | |
| `model` | string | e.g. `gru`, `lstm`, `tcn` |

**Benchmark output (CSV/JSON)**

| field | type | notes |
|-------|------|-------|
| `run_id` | string | UUID per benchmark run |
| `model` | string | |
| `model_size` | string | e.g. `small`, `medium` |
| `mode` | string | `eager` / `tf_function` / `xla` |
| `phase` | string | `train` / `inference` |
| `wall_clock_seconds` | float | |
| `throughput_samples_per_sec` | float | |
| `compile_overhead_seconds` | float | first-call only; null for eager |
| `peak_memory_mb` | float | |

## Coverage

CI enforces a coverage gate of `--cov-fail-under=90`. Coverage is currently 100% across all `kasoku/` modules, validated with shared fixtures across all three execution modes.

## Changelog

### 0.0.1
- Initial release
- GRU, LSTM, and temporal conv models runnable in eager, `tf.function`, and XLA modes
- Data pipeline: windowing, train/val/test split, normalization
- Benchmark suite measuring wall-clock time, throughput, compile overhead, and memory
- Forecaster for point and multi-step inference
- Output sinks: CSV, JSON, optional InfluxDB
- CLI: `train`, `forecast`, `benchmark`
- CI coverage gate enforced at 90%
