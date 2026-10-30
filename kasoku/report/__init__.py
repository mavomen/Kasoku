"""Interactive HTML benchmark reports.

Public API:

- :func:`generate_report` &mdash; one-shot HTML generation
- :func:`watch_and_regenerate` &mdash; live watch mode
- :func:`load_benchmark_data` &mdash; CSV/JSON/InfluxDB loader
- :class:`ReportError` &mdash; report-domain error
"""

from kasoku.report.data import BENCHMARK_COLUMNS, load_benchmark_data
from kasoku.report.errors import ReportError
from kasoku.report.generator import build_html, compute_xla_speedup, generate_report
from kasoku.report.watcher import watch_and_regenerate

__all__ = [
    "BENCHMARK_COLUMNS",
    "ReportError",
    "build_html",
    "compute_xla_speedup",
    "generate_report",
    "load_benchmark_data",
    "watch_and_regenerate",
]
