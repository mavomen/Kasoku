"""Baseline tracking for Kasoku benchmark results.

Loads and saves historical baseline benchmark values (a committed JSON file)
and compares new benchmark runs against them to detect performance
regressions over time.

One baseline entry is stored per unique ``(model, model_size, mode, phase)``
cell, keyed as ``model/size/mode/phase``. For ``xla``/``inference`` cells the
derived SLI ratios (computed against the matching ``eager``/``inference``
cell) are stored as well.

Regression semantics:

* raw metrics ``wall_clock_seconds``, ``compile_overhead_seconds``,
  ``peak_memory_mb``: higher than baseline = regression
* raw metric ``throughput_samples_per_sec``: lower than baseline = regression
* ratio ``xla_speedup_ratio`` (eager/xla): lower than baseline = regression
* ratio ``compile_overhead_ratio``: higher than baseline = regression
* ratio ``throughput_ratio`` (xla/eager): lower than baseline = regression
* ratio ``memory_ratio`` (xla/eager): higher than baseline = regression
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kasoku.config import BaselineConfig, ConfigError

REQUIRED_COLUMNS = {
    "model",
    "model_size",
    "mode",
    "phase",
    "wall_clock_seconds",
    "throughput_samples_per_sec",
    "compile_overhead_seconds",
    "peak_memory_mb",
}

RAW_METRICS = (
    "wall_clock_seconds",
    "throughput_samples_per_sec",
    "compile_overhead_seconds",
    "peak_memory_mb",
)

# (metric, higher value is a regression)
RAW_HIGHER_IS_REGRESSION = frozenset(
    {"wall_clock_seconds", "compile_overhead_seconds", "peak_memory_mb"}
)

RATIO_METRICS = (
    "xla_speedup_ratio",
    "compile_overhead_ratio",
    "throughput_ratio",
    "memory_ratio",
)

# (metric, higher value is a regression)
RATIO_HIGHER_IS_REGRESSION = frozenset(
    {"compile_overhead_ratio", "memory_ratio"}
)

INFERENCE_PHASE = "inference"
XLA_MODE = "xla"
EAGER_MODE = "eager"

DEFAULT_BASELINE_PATH = "benchmarks/baselines.json"


class BaselineError(ConfigError):
    """Raised when baseline loading, saving, or comparison fails."""


def _cell_key(model: str, model_size: str, mode: str, phase: str) -> str:
    return f"{model}/{model_size}/{mode}/{phase}"


def _parse_cells(frame: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """Group frame rows by (model, model_size, mode, phase)."""
    result: dict[tuple, pd.DataFrame] = {}
    for key, group in frame.groupby(
        ["model", "model_size", "mode", "phase"], sort=True
    ):
        result[key] = group
    return result


def _cell_value(
    groups: dict[tuple, pd.DataFrame],
    model: str,
    model_size: str,
    mode: str,
    phase: str,
    column: str,
) -> float | None:
    """Mean of ``column`` for a cell, or None when the cell/value is absent."""
    group = groups.get((model, model_size, mode, phase))
    if group is None:
        return None
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _regression_check(
    current: float | None,
    baseline_value: float | None,
    pct: float,
    higher_is_regression: bool,
) -> tuple[bool | None, float | None]:
    """Return ``(passed, pct_change)``.

    ``passed`` is ``None`` when either value is missing or the baseline is
    zero (cannot evaluate). Otherwise ``True`` means no regression.
    """
    if current is None or baseline_value is None:
        return None, None
    if baseline_value == 0:
        return None, None
    pct_change = (current - baseline_value) / abs(baseline_value) * 100.0
    if higher_is_regression:
        return pct_change <= pct, pct_change
    return pct_change >= -pct, pct_change


def load_baselines(path: str | Path) -> dict:
    """Load baseline JSON from ``path``.

    Returns an empty dict when the file does not exist. Raises
    :class:`BaselineError` on invalid JSON or a non-object root.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"invalid baseline JSON in {p}: {exc}") from exc
    except OSError as exc:
        raise BaselineError(f"cannot read baseline file {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise BaselineError(
            f"baseline root must be a JSON object, got {type(data).__name__}"
        )
    return data


def save_baselines(frame: pd.DataFrame, path: str | Path) -> dict:
    """Write baseline JSON from a benchmark results frame.

    Returns the baseline dict that was written. One entry per cell stores the
    raw metric means; ``xla``/``inference`` cells additionally store the
    derived SLI ratios computed against the matching ``eager``/``inference``
    cell.
    """
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise BaselineError(f"benchmark frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise BaselineError("cannot save baselines from an empty frame")

    groups = _parse_cells(frame)
    baselines: dict[str, dict] = {}

    for (model, model_size, mode, phase), group in groups.items():
        key = _cell_key(model, model_size, mode, phase)
        entry: dict[str, float] = {}
        for column in RAW_METRICS:
            value = _cell_value(groups, model, model_size, mode, phase, column)
            if value is not None:
                entry[column] = value
        baselines[key] = entry

    # Derived SLI ratios for xla/inference cells vs the eager/inference cell.
    for (model, model_size, mode, phase), _group in groups.items():
        if mode != XLA_MODE or phase != INFERENCE_PHASE:
            continue
        key = _cell_key(model, model_size, mode, phase)
        eager_wc = _cell_value(groups, model, model_size, EAGER_MODE, INFERENCE_PHASE, "wall_clock_seconds")
        xla_wc = _cell_value(groups, model, model_size, mode, phase, "wall_clock_seconds")
        if eager_wc is not None and xla_wc is not None and xla_wc > 0:
            baselines[key].setdefault("xla_speedup_ratio", eager_wc / xla_wc)

        xla_co = _cell_value(groups, model, model_size, mode, phase, "compile_overhead_seconds")
        if xla_co is not None and xla_wc is not None and xla_wc > 0:
            baselines[key].setdefault("compile_overhead_ratio", xla_co / xla_wc)

        xla_tp = _cell_value(groups, model, model_size, mode, phase, "throughput_samples_per_sec")
        eager_tp = _cell_value(groups, model, model_size, EAGER_MODE, INFERENCE_PHASE, "throughput_samples_per_sec")
        if xla_tp is not None and eager_tp is not None and eager_tp > 0:
            baselines[key].setdefault("throughput_ratio", xla_tp / eager_tp)

        xla_mem = _cell_value(groups, model, model_size, mode, phase, "peak_memory_mb")
        eager_mem = _cell_value(groups, model, model_size, EAGER_MODE, INFERENCE_PHASE, "peak_memory_mb")
        if xla_mem is not None and eager_mem is not None and eager_mem > 0:
            baselines[key].setdefault("memory_ratio", xla_mem / eager_mem)

    Path(path).write_text(
        json.dumps(baselines, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return baselines


def compare_baselines(
    frame: pd.DataFrame, baselines: dict, config: BaselineConfig
) -> dict:
    """Compare a benchmark frame against stored baselines.

    Returns a dict with::

        {
            "passed": bool,                      # overall, False on any regression
            "cells": {                           # keyed by model/size/mode/phase
                "gru/small/xla/inference": {
                    "wall_clock_seconds": {
                        "passed": bool | None,
                        "baseline": float | None,
                        "current": float | None,
                        "pct_change": float | None,
                        "threshold": float,
                        "regression": bool | None,
                    },
                    ...
                },
            },
            "missing_baselines": [str],          # cells with no stored baseline
        }

    Cells with no stored baseline pass with a warning (recorded in
    ``missing_baselines``). Disabled metrics are skipped.
    """
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise BaselineError(f"benchmark frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise BaselineError("cannot compare an empty benchmark frame")

    groups = _parse_cells(frame)
    cell_results: dict[str, dict] = {}
    missing_baselines: list[str] = []
    overall_passed = True

    target_map = {
        "wall_clock_seconds": config.wall_clock_seconds,
        "throughput_samples_per_sec": config.throughput_samples_per_sec,
        "compile_overhead_seconds": config.compile_overhead_seconds,
        "peak_memory_mb": config.peak_memory_mb,
        "xla_speedup_ratio": config.xla_speedup_ratio,
        "compile_overhead_ratio": config.compile_overhead_ratio,
        "throughput_ratio": config.throughput_ratio,
        "memory_ratio": config.memory_ratio,
    }

    for (model, model_size, mode, phase), group in groups.items():
        key = _cell_key(model, model_size, mode, phase)
        baseline_entry = baselines.get(key)
        cell_metrics: dict[str, dict] = {}

        if baseline_entry is None:
            missing_baselines.append(key)
            cell_results[key] = cell_metrics
            continue

        cell_passed = True

        # Raw metrics compared cell-to-cell.
        for column in RAW_METRICS:
            target_cfg = target_map[column]
            if not target_cfg.enabled:
                continue
            current = _cell_value(groups, model, model_size, mode, phase, column)
            baseline_value = baseline_entry.get(column)
            higher = column in RAW_HIGHER_IS_REGRESSION
            passed, pct_change = _regression_check(
                current, baseline_value, target_cfg.pct, higher
            )
            regression = None if passed is None else not passed
            if regression is True:
                cell_passed = False
                overall_passed = False
            cell_metrics[column] = {
                "passed": passed,
                "baseline": baseline_value,
                "current": current,
                "pct_change": pct_change,
                "threshold": target_cfg.pct,
                "regression": regression,
            }

        # Derived ratios only meaningful for xla/inference cells.
        if mode == XLA_MODE and phase == INFERENCE_PHASE:
            eager_wc = _cell_value(
                groups, model, model_size, EAGER_MODE, INFERENCE_PHASE, "wall_clock_seconds"
            )
            xla_wc = _cell_value(groups, model, model_size, mode, phase, "wall_clock_seconds")
            xla_co = _cell_value(
                groups, model, model_size, mode, phase, "compile_overhead_seconds"
            )
            xla_tp = _cell_value(
                groups, model, model_size, mode, phase, "throughput_samples_per_sec"
            )
            eager_tp = _cell_value(
                groups, model, model_size, EAGER_MODE, INFERENCE_PHASE,
                "throughput_samples_per_sec",
            )
            xla_mem = _cell_value(
                groups, model, model_size, mode, phase, "peak_memory_mb"
            )
            eager_mem = _cell_value(
                groups, model, model_size, EAGER_MODE, INFERENCE_PHASE,
                "peak_memory_mb",
            )

            ratio_specs = (
                ("xla_speedup_ratio", eager_wc / xla_wc if eager_wc is not None and xla_wc is not None and xla_wc > 0 else None),
                ("compile_overhead_ratio", xla_co / xla_wc if xla_co is not None and xla_wc is not None and xla_wc > 0 else None),
                ("throughput_ratio", xla_tp / eager_tp if xla_tp is not None and eager_tp is not None and eager_tp > 0 else None),
                ("memory_ratio", xla_mem / eager_mem if xla_mem is not None and eager_mem is not None and eager_mem > 0 else None),
            )
            for ratio_name, current in ratio_specs:
                target_cfg = target_map[ratio_name]
                if not target_cfg.enabled:
                    continue
                baseline_value = baseline_entry.get(ratio_name)
                higher = ratio_name in RATIO_HIGHER_IS_REGRESSION
                passed, pct_change = _regression_check(
                    current, baseline_value, target_cfg.pct, higher
                )
                regression = None if passed is None else not passed
                if regression is True:
                    cell_passed = False
                    overall_passed = False
                cell_metrics[ratio_name] = {
                    "passed": passed,
                    "baseline": baseline_value,
                    "current": current,
                    "pct_change": pct_change,
                    "threshold": target_cfg.pct,
                    "regression": regression,
                }

        cell_results[key] = cell_metrics
        if not cell_passed:
            overall_passed = False

    return {
        "passed": overall_passed,
        "cells": cell_results,
        "missing_baselines": missing_baselines,
    }


__all__ = [
    "BaselineError",
    "compare_baselines",
    "load_baselines",
    "save_baselines",
    "DEFAULT_BASELINE_PATH",
    "INFERENCE_PHASE",
    "XLA_MODE",
    "EAGER_MODE",
]
