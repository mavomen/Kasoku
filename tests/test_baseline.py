"""Unit tests for baseline tracking."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from kasoku.baseline import (
    BaselineError,
    compare_baselines,
    load_baselines,
    save_baselines,
)
from kasoku.config import BaselineConfig, ConfigError, parse_config


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


def _write_csv(tmp_path, frame) -> str:
    path = tmp_path / "bench.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_save_and_load_roundtrip(tmp_path):
    out = tmp_path / "baselines.json"
    save_baselines(_frame(), str(out))
    loaded = load_baselines(str(out))
    assert "gru/small/eager/inference" in loaded
    assert "gru/small/xla/inference" in loaded
    assert loaded["gru/small/xla/inference"]["wall_clock_seconds"] == 1.0
    # derived ratios stored on xla/inference cell
    xla = loaded["gru/small/xla/inference"]
    assert xla["xla_speedup_ratio"] == 2.0
    assert xla["compile_overhead_ratio"] == 0.2
    assert xla["throughput_ratio"] == 2.0
    assert xla["memory_ratio"] == 1.05
    # raw json is valid
    with out.open() as f:
        json.load(f)


def test_load_missing_file_returns_empty():
    assert load_baselines("/nonexistent/path.json") == {}


def test_load_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(BaselineError, match="invalid baseline JSON"):
        load_baselines(str(bad))


def test_save_missing_columns_raises(tmp_path):
    frame = _frame()[["model", "mode"]]
    with pytest.raises(BaselineError, match="missing columns"):
        save_baselines(frame, str(tmp_path / "b.json"))


def test_save_empty_frame_raises(tmp_path):
    with pytest.raises(BaselineError, match="empty frame"):
        save_baselines(_frame().iloc[0:0], str(tmp_path / "b.json"))


def test_compare_identical_passes():
    baselines = save_baselines(_frame(), "/tmp/kilo/t_identical.json")
    results = compare_baselines(_frame(), baselines, BaselineConfig())
    assert results["passed"] is True
    assert results["missing_baselines"] == []


def test_compare_wall_clock_regression():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "wall_clock_seconds"] = 1.5  # +50%
    baselines = save_baselines(_frame(), "/tmp/kilo/t_wc.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    assert results["passed"] is False
    entry = results["cells"]["gru/small/xla/inference"]["wall_clock_seconds"]
    assert entry["regression"] is True
    assert entry["pct_change"] == pytest.approx(50.0)


def test_compare_throughput_regression():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "throughput_samples_per_sec"] = 5.0  # -75% (baseline 20)
    baselines = save_baselines(_frame(), "/tmp/kilo/t_tp.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    assert results["passed"] is False
    assert results["cells"]["gru/small/xla/inference"]["throughput_samples_per_sec"]["regression"] is True


def test_compare_compile_overhead_regression():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "compile_overhead_seconds"] = 0.5  # +150% (baseline 0.2)
    baselines = save_baselines(_frame(), "/tmp/kilo/t_co.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    assert results["passed"] is False
    assert results["cells"]["gru/small/xla/inference"]["compile_overhead_seconds"]["regression"] is True


def test_compare_memory_regression():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "peak_memory_mb"] = 200.0  # ~+90% (baseline 105)
    baselines = save_baselines(_frame(), "/tmp/kilo/t_mem.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    assert results["passed"] is False
    assert results["cells"]["gru/small/xla/inference"]["peak_memory_mb"]["regression"] is True


def test_compare_speedup_ratio_regression():
    frame = _frame()
    frame.loc[frame["mode"] == "xla", "wall_clock_seconds"] = 1.8  # speedup 2.0 -> 1.11 (-44%)
    baselines = save_baselines(_frame(), "/tmp/kilo/t_sp.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    assert results["passed"] is False
    assert (
        results["cells"]["gru/small/xla/inference"]["xla_speedup_ratio"]["regression"]
        is True
    )


def test_compare_missing_baseline_passes_with_warning():
    results = compare_baselines(_frame(), {}, BaselineConfig())
    assert results["passed"] is True
    assert results["missing_baselines"] == [
        "gru/small/eager/inference",
        "gru/small/xla/inference",
    ]


def test_compare_disabled_metrics_skipped():
    config = BaselineConfig()
    config.wall_clock_seconds.enabled = False
    # data identical to baseline so nothing else regresses
    baselines = save_baselines(_frame(), "/tmp/kilo/t_disabled.json")
    results = compare_baselines(_frame(), baselines, config)
    cell = results["cells"]["gru/small/xla/inference"]
    # disabled metric is not evaluated: absent from cell metrics
    assert "wall_clock_seconds" not in cell
    # overall still passes because the disabled metric is not evaluated
    assert results["passed"] is True


def test_compare_empty_frame_raises():
    with pytest.raises(BaselineError, match="empty benchmark frame"):
        compare_baselines(_frame().iloc[0:0], {}, BaselineConfig())


def test_compare_multiple_combos():
    frame = pd.concat(
        [
            _frame(),
            pd.DataFrame(
                {
                    "run_id": ["run-1"] * 4,
                    "model": ["gru", "gru", "gru", "gru"],
                    "model_size": ["medium", "medium", "medium", "medium"],
                    "mode": ["eager", "xla", "eager", "xla"],
                    "phase": ["inference"] * 4,
                    "wall_clock_seconds": [4.0, 3.0, 4.0, 3.0],
                    "throughput_samples_per_sec": [5.0, 5.0, 5.0, 5.0],
                    "compile_overhead_seconds": [float("nan"), 0.1, float("nan"), 0.1],
                    "peak_memory_mb": [200.0, 210.0, 200.0, 210.0],
                }
            ),
        ],
        ignore_index=True,
    )
    baselines = save_baselines(frame, "/tmp/kilo/t_multi.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    assert results["passed"] is True
    assert set(results["cells"]) == {
        "gru/medium/eager/inference",
        "gru/medium/xla/inference",
        "gru/small/eager/inference",
        "gru/small/xla/inference",
    }


def test_compare_missing_eager_derives_none():
    # No eager cell: derived ratios cannot be computed and are skipped.
    frame = _frame()[_frame()["mode"] == "xla"]
    baselines = save_baselines(frame, "/tmp/kilo/t_noeager.json")
    results = compare_baselines(frame, baselines, BaselineConfig())
    cell = results["cells"]["gru/small/xla/inference"]
    assert cell["xla_speedup_ratio"]["regression"] is None
    assert results["passed"] is True


def test_baseline_config_defaults():
    config = BaselineConfig()
    assert config.wall_clock_seconds.pct == 10.0
    assert config.wall_clock_seconds.enabled is True
    assert config.compile_overhead_seconds.pct == 20.0
    assert config.memory_ratio.pct == 10.0


def test_baseline_config_validation_pct_must_be_positive():
    with pytest.raises(ConfigError, match="pct must be"):
        parse_config({"baseline": {"wall_clock_seconds": {"pct": 0}}})
    with pytest.raises(ConfigError, match="pct must be"):
        parse_config({"baseline": {"wall_clock_seconds": {"pct": -5}}})


def test_baseline_config_partial_merge_with_defaults(tmp_path):
    import tomllib

    from kasoku.config import load_config

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[baseline]\nwall_clock_seconds = { pct = 25 }\n', encoding="utf-8"
    )
    config = load_config(cfg)
    assert config.baseline.wall_clock_seconds.pct == 25.0
    assert config.baseline.throughput_samples_per_sec.pct == 10.0
    assert config.baseline.compile_overhead_seconds.pct == 20.0
