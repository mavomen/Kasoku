"""HTML benchmark report generation entry point.

Loads benchmark data from CSV/JSON/InfluxDB, computes derived metrics (XLA
speedup per model/size), and renders a self-contained HTML dashboard with
embedded Chart.js charts and HTMX-powered interactivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kasoku.config import Config
from kasoku.report.data import load_accuracy_data, load_benchmark_data
from kasoku.report.errors import ReportError
from kasoku.report.templates import REPORT_TEMPLATE


def _sanitize_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame to JSON-safe records (NaN/NaT -> None)."""
    records: list[dict] = []
    for _, row in frame.iterrows():
        record: dict = {}
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                record[column] = None
            elif isinstance(value, (pd.Timestamp,)):
                record[column] = value.isoformat()
            else:
                record[column] = value
        records.append(record)
    return records


def compute_xla_speedup(frame: pd.DataFrame) -> list[dict]:
    """Compute XLA speedup (eager wall-clock / XLA wall-clock) per model/size.

    Only the inference phase is considered; cells without both an eager and an
    XLA measurement, or with a non-positive XLA wall-clock, are skipped.
    """
    inference = frame[frame["phase"] == "inference"]
    speedups: list[dict] = []
    for (model, model_size), group in inference.groupby(["model", "model_size"], sort=True):
        eager = group.loc[group["mode"] == "eager", "wall_clock_seconds"]
        xla = group.loc[group["mode"] == "xla", "wall_clock_seconds"]
        if eager.empty or xla.empty:
            continue
        eager_wc = float(eager.iloc[0])
        xla_wc = float(xla.iloc[0])
        if xla_wc <= 0:
            continue
        speedups.append({"model": model, "model_size": model_size, "speedup": eager_wc / xla_wc})
    return speedups


def build_html(
    frame: pd.DataFrame,
    compare_frame: pd.DataFrame | None = None,
    accuracy_data: dict | None = None,
    compare_accuracy_data: dict | None = None,
) -> str:
    """Render the self-contained HTML report from a benchmark frame.

    Optionally injects accuracy data (from ``kasoku accuracy check --output``)
    so the dashboard can render MAE/MAPE per-series charts with threshold
    lines and pass/fail status.
    """
    data_json = json.dumps(_sanitize_records(frame), default=str)
    if compare_frame is None:
        compare_json = json.dumps([])
    else:
        compare_json = json.dumps(_sanitize_records(compare_frame), default=str)
    speedup_json = json.dumps(compute_xla_speedup(frame))
    if accuracy_data is None:
        accuracy_json = json.dumps({})
    else:
        accuracy_json = json.dumps(accuracy_data, default=str)
    if compare_accuracy_data is None:
        compare_accuracy_json = json.dumps({})
    else:
        compare_accuracy_json = json.dumps(compare_accuracy_data, default=str)
    return (
        REPORT_TEMPLATE
        .replace("__BENCHMARK_DATA__", data_json)
        .replace("__COMPARE_DATA__", compare_json)
        .replace("__SPEEDUP_DATA__", speedup_json)
        .replace("__ACCURACY_DATA__", accuracy_json)
        .replace("__COMPARE_ACCURACY_DATA__", compare_accuracy_json)
    )


def generate_report(
    data: str,
    output: str,
    *,
    compare: str | None = None,
    config: Config | None = None,
    time_range: str = "-24h",
    accuracy: str | None = None,
    compare_accuracy: str | None = None,
) -> str:
    """Generate a self-contained HTML benchmark report and write it to ``output``.

    Returns the output path.
    """
    frame = load_benchmark_data(data, config=config, time_range=time_range)
    if frame.empty:
        raise ReportError("benchmark data is empty; cannot generate report")
    compare_frame: pd.DataFrame | None = None
    if compare is not None:
        compare_frame = load_benchmark_data(compare, config=config, time_range=time_range)
        if compare_frame.empty:
            raise ReportError("compare benchmark data is empty")
    accuracy_data: dict | None = None
    if accuracy is not None:
        accuracy_data = load_accuracy_data(accuracy)
    compare_accuracy_data: dict | None = None
    if compare_accuracy is not None:
        compare_accuracy_data = load_accuracy_data(compare_accuracy)
    html = build_html(frame, compare_frame, accuracy_data, compare_accuracy_data)
    Path(output).write_text(html, encoding="utf-8")
    return output
