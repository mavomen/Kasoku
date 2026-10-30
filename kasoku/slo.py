"""Service Level Objective (SLO) evaluation for Kasoku benchmark results.

Computes Service Level Indicators (SLIs) from a benchmark results frame and
evaluates them against configurable SLO targets, returning pass/fail per SLO.
SLIs compare XLA execution mode against eager baseline within the inference
phase, grouped by model and model size.
"""

from __future__ import annotations

import pandas as pd

from kasoku.config import ConfigError, SLOConfig

INFERENCE_PHASE = "inference"
XLA_MODE = "xla"
EAGER_MODE = "eager"

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


class SLOError(ConfigError):
    """Raised when SLO evaluation fails."""


def _lookup(
    frame: pd.DataFrame,
    model: str,
    model_size: str,
    mode: str,
    phase: str,
    column: str,
) -> float | None:
    mask = (
        (frame["model"] == model)
        & (frame["model_size"] == model_size)
        & (frame["mode"] == mode)
        & (frame["phase"] == phase)
    )
    matches = frame.loc[mask, column]
    if matches.empty:
        return None
    return float(matches.iloc[0])


def _check(value: float | None, target: float, operator: str) -> bool | None:
    if value is None:
        return None
    if operator == ">=":
        return value >= target
    return value <= target


def evaluate_slo(frame: pd.DataFrame, config: SLOConfig) -> dict[str, dict]:
    """Evaluate SLO targets against a benchmark results frame.

    Returns a dict keyed by SLO name with entries:

    ``{"passed": bool, "value": float | None, "target": float,
      "operator": str, "combos": dict[str, dict]}``

    An SLO passes only when every model/size combo passes. ``value`` is the
    worst-case across combos (minimum for ``>=`` targets, maximum for ``<=``
    targets). ``combos`` lists per-combo value and pass/fail.
    """
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise SLOError(f"benchmark frame missing columns: {sorted(missing)}")

    inf = frame[frame["phase"] == INFERENCE_PHASE]
    if inf.empty:
        raise SLOError("no inference-phase rows in benchmark frame")

    combos = inf[["model", "model_size"]].drop_duplicates()
    if combos.empty:
        raise SLOError("no model/size combos in inference-phase rows")

    slo_specs = (
        ("xla_speedup", config.xla_speedup, _sli_xla_speedup),
        ("compile_overhead", config.compile_overhead, _sli_compile_overhead),
        ("throughput_ratio", config.throughput_ratio, _sli_throughput_ratio),
        ("memory_ratio", config.memory_ratio, _sli_memory_ratio),
    )

    results: dict[str, dict] = {}
    for name, target_cfg, sli_fn in slo_specs:
        if not target_cfg.enabled:
            results[name] = {
                "passed": True,
                "value": None,
                "target": target_cfg.target,
                "operator": target_cfg.operator,
                "combos": {},
            }
            continue

        combo_results: dict[str, dict] = {}
        all_passed = True
        worst_value: float | None = None

        for _, combo in combos.iterrows():
            model = str(combo["model"])
            model_size = str(combo["model_size"])
            key = f"{model}/{model_size}"

            value = sli_fn(inf, model, model_size)
            passed = _check(value, target_cfg.target, target_cfg.operator)
            combo_results[key] = {"value": value, "passed": passed}

            if passed is False:
                all_passed = False

            if value is not None:
                if target_cfg.operator == ">=":
                    if worst_value is None or value < worst_value:
                        worst_value = value
                else:
                    if worst_value is None or value > worst_value:
                        worst_value = value

        results[name] = {
            "passed": all_passed,
            "value": worst_value,
            "target": target_cfg.target,
            "operator": target_cfg.operator,
            "combos": combo_results,
        }

    return results


def _sli_xla_speedup(
    frame: pd.DataFrame, model: str, model_size: str
) -> float | None:
    """XLA speedup ratio = eager wall-clock / XLA wall-clock (>= 1 means faster)."""
    eager_wc = _lookup(frame, model, model_size, EAGER_MODE, INFERENCE_PHASE, "wall_clock_seconds")
    xla_wc = _lookup(frame, model, model_size, XLA_MODE, INFERENCE_PHASE, "wall_clock_seconds")
    if eager_wc is None or xla_wc is None or xla_wc <= 0:
        return None
    return eager_wc / xla_wc


def _sli_compile_overhead(
    frame: pd.DataFrame, model: str, model_size: str
) -> float | None:
    """Compile overhead ratio = compile overhead / XLA wall-clock."""
    xla_co = _lookup(
        frame, model, model_size, XLA_MODE, INFERENCE_PHASE, "compile_overhead_seconds"
    )
    xla_wc = _lookup(frame, model, model_size, XLA_MODE, INFERENCE_PHASE, "wall_clock_seconds")
    if xla_co is None or xla_wc is None or xla_wc <= 0:
        return None
    return xla_co / xla_wc


def _sli_throughput_ratio(
    frame: pd.DataFrame, model: str, model_size: str
) -> float | None:
    """Throughput ratio = XLA throughput / eager throughput (>= 1 means higher)."""
    xla_tp = _lookup(
        frame, model, model_size, XLA_MODE, INFERENCE_PHASE, "throughput_samples_per_sec"
    )
    eager_tp = _lookup(
        frame, model, model_size, EAGER_MODE, INFERENCE_PHASE, "throughput_samples_per_sec"
    )
    if xla_tp is None or eager_tp is None or eager_tp <= 0:
        return None
    return xla_tp / eager_tp


def _sli_memory_ratio(
    frame: pd.DataFrame, model: str, model_size: str
) -> float | None:
    """Memory ratio = XLA peak memory / eager peak memory."""
    xla_mem = _lookup(
        frame, model, model_size, XLA_MODE, INFERENCE_PHASE, "peak_memory_mb"
    )
    eager_mem = _lookup(
        frame, model, model_size, EAGER_MODE, INFERENCE_PHASE, "peak_memory_mb"
    )
    if xla_mem is None or eager_mem is None or eager_mem <= 0:
        return None
    return xla_mem / eager_mem


def all_passed(results: dict[str, dict]) -> bool:
    """Return True if every SLO result passed."""
    return all(entry["passed"] for entry in results.values())


__all__ = [
    "SLOError",
    "evaluate_slo",
    "all_passed",
    "INFERENCE_PHASE",
    "XLA_MODE",
    "EAGER_MODE",
]
