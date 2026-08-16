"""Unit tests for benchmark runner sweep completeness and output schema."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from kasoku.benchmark import (
    BENCHMARK_COLUMNS,
    BenchmarkError,
    aggregate_results,
    run_benchmark,
)
from kasoku.data import CsvSource
from kasoku.pipeline import PipelineError

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"

MODELS = ["gru", "lstm", "tcn"]
MODES = ["eager", "tf_function", "xla"]
PHASES = ["train", "inference"]

UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def _fast_kwargs(**overrides):
    kwargs = dict(
        lookback=8,
        horizon=1,
        batch_size=16,
        iterations=2,
        warmup_iterations=1,
    )
    kwargs.update(overrides)
    return kwargs


def _series():
    return CsvSource(SINGLE).read()


@pytest.mark.parametrize("model", MODELS)
def test_benchmark_runner_smoke_every_model_mode_combination(model):
    df = _series()
    frame = run_benchmark(df, model, sizes=["small"], modes=MODES, phases=PHASES, **_fast_kwargs(iterations=1, warmup_iterations=0, batch_size=8))
    assert len(frame) == len(MODES) * len(PHASES)
    assert set(frame["mode"]) == set(MODES)
    assert set(frame["phase"]) == set(PHASES)
    assert (frame["wall_clock_seconds"] > 0).all()
    assert (frame["throughput_samples_per_sec"] > 0).all()
    assert (frame["peak_memory_mb"] >= 0).all()
    assert frame["run_id"].nunique() == 1


def test_run_benchmark_sweep_completeness():
    sizes, modes, phases = ["small", "medium"], ["eager", "tf_function"], PHASES
    frame = run_benchmark(_series(), "gru", sizes=sizes, modes=modes, phases=phases, **_fast_kwargs())
    assert len(frame) == len(sizes) * len(modes) * len(phases)
    combos = set(zip(frame["model_size"], frame["mode"], frame["phase"]))
    expected = {
        (s, m, p)
        for s in sizes
        for m in modes
        for p in phases
    }
    assert combos == expected


def test_run_benchmark_output_schema_columns_and_values():
    frame = run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases=PHASES, **_fast_kwargs())
    assert list(frame.columns) == BENCHMARK_COLUMNS
    assert frame["model"].eq("gru").all()
    assert frame["model_size"].eq("small").all()
    assert frame["mode"].eq("eager").all()
    assert frame["phase"].tolist() == PHASES
    assert (frame["wall_clock_seconds"] > 0).all()
    assert (frame["throughput_samples_per_sec"] > 0).all()
    assert frame["compile_overhead_seconds"].isna().all()
    assert (frame["peak_memory_mb"] >= 0).all()
    assert re.fullmatch(UUID_RE, frame["run_id"].iloc[0])


def test_run_benchmark_graph_mode_reports_compile_overhead():
    frame = run_benchmark(_series(), "gru", sizes=["small"], modes=["tf_function"], phases=["inference"], **_fast_kwargs())
    overhead = frame["compile_overhead_seconds"].iloc[0]
    assert pd.notna(overhead)
    assert overhead >= 0.0


def test_run_benchmark_throughput_consistent_with_wall_clock():
    frame = run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases=["inference"], **_fast_kwargs())
    row = frame.iloc[0]
    assert row["throughput_samples_per_sec"] == pytest.approx(16 / row["wall_clock_seconds"])


def test_run_benchmark_run_id_shared_within_run_and_unique_across_runs():
    kwargs = _fast_kwargs(sizes=["small", "medium"], modes=["eager"])
    frame = run_benchmark(_series(), "gru", **kwargs)
    assert frame["run_id"].nunique() == 1
    other = run_benchmark(_series(), "gru", **kwargs)
    assert frame["run_id"].iloc[0] != other["run_id"].iloc[0]


def test_run_benchmark_unknown_model_raises():
    with pytest.raises(BenchmarkError, match="unknown model"):
        run_benchmark(_series(), "transformer", sizes=["small"], modes=["eager"], phases=["inference"], **_fast_kwargs())


def test_run_benchmark_unknown_size_raises():
    with pytest.raises(BenchmarkError, match="unknown gru size"):
        run_benchmark(_series(), "gru", sizes=["huge"], modes=["eager"], phases=["inference"], **_fast_kwargs())


def test_run_benchmark_unknown_mode_raises():
    with pytest.raises(BenchmarkError, match="unknown execution mode"):
        run_benchmark(_series(), "gru", sizes=["small"], modes=["quantum"], phases=["inference"], **_fast_kwargs())


def test_run_benchmark_unknown_phase_raises():
    with pytest.raises(BenchmarkError, match="unknown benchmark phase"):
        run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases=["eval"], **_fast_kwargs())


def test_run_benchmark_validates_batch_size():
    with pytest.raises(BenchmarkError, match="batch_size"):
        run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases=["inference"], **_fast_kwargs(batch_size=0))


def test_run_benchmark_validates_iterations():
    with pytest.raises(BenchmarkError, match="iterations"):
        run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases=["inference"], **_fast_kwargs(iterations=0))


def test_run_benchmark_validates_warmup_iterations():
    with pytest.raises(BenchmarkError, match="warmup_iterations"):
        run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases=["inference"], **_fast_kwargs(warmup_iterations=-1))


def test_run_benchmark_rejects_string_lists():
    with pytest.raises(BenchmarkError, match="sizes"):
        run_benchmark(_series(), "gru", sizes="small", modes=["eager"], phases=["inference"], **_fast_kwargs())
    with pytest.raises(BenchmarkError, match="modes"):
        run_benchmark(_series(), "gru", sizes=["small"], modes="eager", phases=["inference"], **_fast_kwargs())
    with pytest.raises(BenchmarkError, match="phases"):
        run_benchmark(_series(), "gru", sizes=["small"], modes=["eager"], phases="inference", **_fast_kwargs())


def test_run_benchmark_non_canonical_df_raises():
    bad = _series().rename(columns={"value": "vals"})
    with pytest.raises(PipelineError):
        run_benchmark(bad, "gru", sizes=["small"], modes=["eager"], phases=["inference"], **_fast_kwargs())


def test_run_benchmark_insufficient_data_raises():
    tiny = _series().head(10)
    with pytest.raises(PipelineError, match="windows"):
        run_benchmark(tiny, "gru", sizes=["small"], modes=["eager"], phases=["inference"], lookback=24, horizon=1, batch_size=8, iterations=1, warmup_iterations=0)


def test_aggregate_results_builds_schema_frame():
    rows = [{
        "run_id": "run-1",
        "model": "gru",
        "model_size": "small",
        "mode": "eager",
        "phase": "inference",
        "wall_clock_seconds": 0.1,
        "throughput_samples_per_sec": 160.0,
        "compile_overhead_seconds": None,
        "peak_memory_mb": 1.0,
    }]
    frame = aggregate_results(rows, "run-1")
    assert list(frame.columns) == BENCHMARK_COLUMNS
    assert len(frame) == 1
    assert frame["run_id"].iloc[0] == "run-1"
    assert pd.isna(frame["compile_overhead_seconds"].iloc[0])


def test_aggregate_results_empty_rows_raises():
    with pytest.raises(BenchmarkError, match="empty"):
        aggregate_results([], "run-1")


def test_aggregate_results_validates_run_id():
    row = {
        "model": "gru",
        "model_size": "small",
        "mode": "eager",
        "phase": "inference",
    }
    with pytest.raises(BenchmarkError, match="run_id"):
        aggregate_results([row], "")
