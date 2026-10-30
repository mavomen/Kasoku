"""Unit tests for SLO evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from kasoku.config import ConfigError, SLOConfig, parse_config
from kasoku.slo import SLOError, all_passed, evaluate_slo


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["run-1"] * 4,
            "model": ["gru", "gru", "gru", "gru"],
            "model_size": ["small", "small", "small", "small"],
            "mode": ["eager", "xla", "eager", "xla"],
            "phase": ["inference", "inference", "inference", "inference"],
            "wall_clock_seconds": [2.0, 1.0, 2.0, 1.0],
            "throughput_samples_per_sec": [10.0, 20.0, 10.0, 20.0],
            "compile_overhead_seconds": [float("nan"), 0.2, float("nan"), 0.2],
            "peak_memory_mb": [100.0, 105.0, 100.0, 105.0],
        }
    )


def test_evaluate_slo_all_pass():
    results = evaluate_slo(_frame(), SLOConfig())
    assert all_passed(results)
    assert results["xla_speedup"]["passed"] is True
    assert results["xla_speedup"]["value"] == 2.0
    assert results["compile_overhead"]["passed"] is True
    assert results["compile_overhead"]["value"] == 0.2
    assert results["throughput_ratio"]["passed"] is True
    assert results["throughput_ratio"]["value"] == 2.0
    assert results["memory_ratio"]["passed"] is True
    assert results["memory_ratio"]["value"] == 1.05


def test_evaluate_slo_speedup_fail():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "wall_clock_seconds"] = 1.8
    results = evaluate_slo(frame, SLOConfig())
    assert results["xla_speedup"]["passed"] is False
    assert not all_passed(results)


def test_evaluate_slo_compile_overhead_fail():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "compile_overhead_seconds"] = 0.5
    results = evaluate_slo(frame, SLOConfig())
    assert results["compile_overhead"]["passed"] is False
    assert not all_passed(results)


def test_evaluate_slo_throughput_fail():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "throughput_samples_per_sec"] = 5.0
    results = evaluate_slo(frame, SLOConfig())
    assert results["throughput_ratio"]["passed"] is False
    assert not all_passed(results)


def test_evaluate_slo_memory_fail():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "peak_memory_mb"] = 150.0
    results = evaluate_slo(frame, SLOConfig())
    assert results["memory_ratio"]["passed"] is False
    assert not all_passed(results)


def test_evaluate_slo_disabled():
    config = SLOConfig()
    config.xla_speedup.enabled = False
    results = evaluate_slo(_frame(), config)
    assert results["xla_speedup"]["passed"] is True
    assert results["xla_speedup"]["value"] is None
    assert results["xla_speedup"]["combos"] == {}


def test_evaluate_slo_missing_columns_raises():
    frame = _frame()[["model", "mode"]]
    with pytest.raises(SLOError, match="missing columns"):
        evaluate_slo(frame, SLOConfig())


def test_evaluate_slo_no_inference_raises():
    frame = _frame()
    frame = frame[frame["phase"] == "train"]
    with pytest.raises(SLOError, match="no inference-phase rows"):
        evaluate_slo(frame, SLOConfig())


def test_evaluate_slo_missing_mode_returns_none():
    frame = _frame()
    frame = frame[frame["mode"] != "xla"]
    results = evaluate_slo(frame, SLOConfig())
    assert results["xla_speedup"]["combos"]["gru/small"]["value"] is None


def test_all_passed_empty():
    assert all_passed({}) is True


def test_slo_config_defaults():
    config = SLOConfig()
    assert config.xla_speedup.target == 1.20
    assert config.xla_speedup.operator == ">="
    assert config.compile_overhead.target == 0.30
    assert config.compile_overhead.operator == "<="
    assert config.throughput_ratio.target == 0.90
    assert config.throughput_ratio.operator == ">="
    assert config.memory_ratio.target == 1.10
    assert config.memory_ratio.operator == "<="


def test_slo_config_validation_invalid_operator():
    with pytest.raises(ConfigError, match="operator must be"):
        parse_config({"slo": {"xla_speedup": {"operator": "=="}}})


def test_multiple_combos():
    frame = pd.concat(
        [
            _frame(),
            pd.DataFrame(
                {
                    "run_id": ["run-1"] * 4,
                    "model": ["gru", "gru", "gru", "gru"],
                    "model_size": ["medium", "medium", "medium", "medium"],
                    "mode": ["eager", "xla", "eager", "xla"],
                    "phase": ["inference", "inference", "inference", "inference"],
                    "wall_clock_seconds": [4.0, 3.5, 4.0, 3.5],
                    "throughput_samples_per_sec": [5.0, 5.0, 5.0, 5.0],
                    "compile_overhead_seconds": [float("nan"), 0.1, float("nan"), 0.1],
                    "peak_memory_mb": [200.0, 210.0, 200.0, 210.0],
                }
            ),
        ],
        ignore_index=True,
    )
    results = evaluate_slo(frame, SLOConfig())
    # medium combo: speedup = 4.0/3.5 = 1.14 < 1.20 -> FAIL
    assert results["xla_speedup"]["combos"]["gru/small"]["passed"] is True
    assert results["xla_speedup"]["combos"]["gru/medium"]["passed"] is False
    assert not all_passed(results)
