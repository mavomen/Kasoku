"""Benchmark runner: sweep model x size x mode x phase.

The runner measures every combination with the generic primitives from
:mod:`kasoku.benchmark.harness`: steady-state wall-clock timing, throughput
(samples per second), first-call compile overhead (isolated from steady-state
and null for eager), and peak resident memory. The train phase measures a
single gradient-descent train step over one batch (reusing the shared
``_train_step`` from the training loop), so every mode trains through the
exact same code path. A fresh model is built per cell with a fixed seed, so
weights are identical across modes and train-step mutations never leak into
later inference cells.
"""

from __future__ import annotations

import uuid
from functools import partial
from typing import Sequence

import pandas as pd
import tensorflow as tf

from kasoku.benchmark.base import BenchmarkError
from kasoku.benchmark.harness import (
    compile_overhead,
    peak_memory_mb,
    resolve_phase,
    throughput,
    time_phase,
)
from kasoku.execution import VALID_MODES, compile_call
from kasoku.models import (
    GRU_SIZES,
    LSTM_SIZES,
    TCN_SIZES,
    build_gru_by_size,
    build_lstm_by_size,
    build_tcn_by_size,
)
from kasoku.pipeline import validate_canonical, window_series
from kasoku.train.trainer import _train_step

BENCHMARK_COLUMNS = [
    "run_id",
    "model",
    "model_size",
    "mode",
    "phase",
    "wall_clock_seconds",
    "throughput_samples_per_sec",
    "compile_overhead_seconds",
    "peak_memory_mb",
]

_BUILDERS = {
    "gru": build_gru_by_size,
    "lstm": build_lstm_by_size,
    "tcn": build_tcn_by_size,
}
_REGISTRIES = {"gru": GRU_SIZES, "lstm": LSTM_SIZES, "tcn": TCN_SIZES}


def aggregate_results(rows: list[dict], run_id: str) -> pd.DataFrame:
    """Assemble benchmark row records into a schema'd results frame.

    The frame columns match :data:`BENCHMARK_COLUMNS` exactly (and therefore
    the AGENTS.md benchmark output schema), so a sink can write it directly.
    """
    if not isinstance(run_id, str) or not run_id:
        raise BenchmarkError("run_id must be a non-empty string")
    if not rows:
        raise BenchmarkError("cannot aggregate an empty benchmark result set")
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def run_benchmark(
    df: pd.DataFrame,
    model_name: str,
    *,
    sizes: Sequence[str] = ("small", "medium"),
    modes: Sequence[str] = ("eager", "tf_function", "xla"),
    phases: Sequence[str] = ("train", "inference"),
    lookback: int = 24,
    horizon: int = 1,
    batch_size: int = 32,
    seed: int = 42,
    iterations: int = 3,
    warmup_iterations: int = 1,
) -> pd.DataFrame:
    """Sweep ``model_name`` across every size x mode x phase combination.

    ``df`` is a canonical long-form series frame (see
    :func:`kasoku.pipeline.validate_canonical`); windows are pooled per series
    and the first ``batch_size`` samples form the benchmark input so each
    invocation processes ``samples_per_call`` samples. Returns one row per
    cell with columns ``BENCHMARK_COLUMNS``; ``run_id`` (a UUID) is shared
    across all rows of this sweep.
    """
    _validate_model(model_name)
    _validate_sizes(model_name, sizes)
    _validate_modes(modes)
    _validate_phases(phases)
    _validate_batch_size(batch_size)
    if iterations < 1:
        raise BenchmarkError(f"iterations must be >= 1, got {iterations}")
    if warmup_iterations < 0:
        raise BenchmarkError(f"warmup_iterations must be >= 0, got {warmup_iterations}")

    x_batch, y_batch = _prepare_data(df, lookback, horizon, batch_size)
    run_id = str(uuid.uuid4())
    rows: list[dict] = []
    for size in sizes:
        for mode in modes:
            for phase in phases:
                model = _BUILDERS[model_name](
                    size, lookback=lookback, horizon=horizon, seed=seed
                )
                if phase == "train":
                    measurements = _measure_train(
                        model, x_batch, y_batch, mode, iterations, warmup_iterations
                    )
                else:
                    measurements = _measure_inference(
                        model, x_batch, mode, iterations, warmup_iterations
                    )
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model_name,
                        "model_size": size,
                        "mode": mode,
                        "phase": phase,
                        **measurements,
                    }
                )
    return aggregate_results(rows, run_id)


def _measure_inference(
    model,
    x_batch,
    mode: str,
    iterations: int,
    warmup_iterations: int,
) -> dict:
    wrapped = compile_call(model, mode)
    wall_clock = time_phase(
        wrapped,
        x_batch,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    return {
        "wall_clock_seconds": wall_clock,
        "throughput_samples_per_sec": throughput(len(x_batch), wall_clock),
        "compile_overhead_seconds": compile_overhead(
            model,
            x_batch,
            mode=mode,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        ),
        "peak_memory_mb": peak_memory_mb(
            wrapped,
            x_batch,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        ),
    }


def _measure_train(
    model,
    x_batch,
    y_batch,
    mode: str,
    iterations: int,
    warmup_iterations: int,
) -> dict:
    optimizer = tf.keras.optimizers.Adam(1e-3)
    step = partial(_train_step, model, optimizer)
    wrapped = compile_call(step, mode)
    wall_clock = time_phase(
        wrapped,
        x_batch,
        y_batch,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    return {
        "wall_clock_seconds": wall_clock,
        "throughput_samples_per_sec": throughput(len(x_batch), wall_clock),
        "compile_overhead_seconds": compile_overhead(
            step,
            x_batch,
            y_batch,
            mode=mode,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        ),
        "peak_memory_mb": peak_memory_mb(
            wrapped,
            x_batch,
            y_batch,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        ),
    }


def _prepare_data(df, lookback: int, horizon: int, batch_size: int):
    validate_canonical(df)
    x, y = window_series(df, lookback=lookback, horizon=horizon)
    return x[:batch_size], y[:batch_size]


def _validate_model(model_name: str) -> None:
    if model_name not in _BUILDERS:
        raise BenchmarkError(
            f"unknown model {model_name!r}; expected one of {sorted(_BUILDERS)}"
        )


def _validate_sizes(model_name: str, sizes: Sequence[str]) -> None:
    _validate_sequence(sizes, "sizes")
    registry = _REGISTRIES[model_name]
    for size in sizes:
        if size not in registry:
            raise BenchmarkError(
                f"unknown {model_name} size {size!r}; expected one of {sorted(registry)}"
            )


def _validate_modes(modes: Sequence[str]) -> None:
    _validate_sequence(modes, "modes")
    for mode in modes:
        if mode not in VALID_MODES:
            raise BenchmarkError(
                f"unknown execution mode {mode!r}; expected one of {sorted(VALID_MODES)}"
            )


def _validate_phases(phases: Sequence[str]) -> None:
    _validate_sequence(phases, "phases")
    for phase in phases:
        resolve_phase(phase)


def _validate_sequence(values: Sequence[str], name: str) -> None:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise BenchmarkError(
            f"{name} must be a sequence of strings, got {type(values).__name__}"
        )


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise BenchmarkError(f"batch_size must be >= 1, got {batch_size}")
