"""Live watch mode for benchmark reports.

Watches a benchmark data file for changes and regenerates the HTML report on
each modification. Uses :mod:`watchdog` when available for efficient
cross-platform file watching, falling back to :mod:`os.path.getmtime` polling.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from kasoku.config import Config
from kasoku.report.errors import ReportError
from kasoku.report.generator import generate_report


def _file_changed(path: str, last_mtime: float | None) -> bool:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    if last_mtime is None:
        return False
    return mtime != last_mtime


def _current_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def watch_and_regenerate(
    data_path: str,
    output_path: str,
    *,
    compare_path: str | None = None,
    config: Config | None = None,
    time_range: str = "-24h",
    interval: int = 5,
) -> None:
    """Watch ``data_path`` for changes and regenerate ``output_path`` on each change.

    The initial report is generated immediately. Afterwards the loop polls
    every ``interval`` seconds (or uses watchdog if installed) and regenerates
    when the data file (or the compare file, when provided) is modified.

    This function runs forever; callers run it in a background thread or
    process.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class _Handler(FileSystemEventHandler):
            def __init__(self, callback):
                self.callback = callback

            def on_modified(self, event):
                if not event.is_directory:
                    self.callback()

        def run_with_watchdog() -> None:
            generate_report(data_path, output_path, compare=compare_path, config=config, time_range=time_range)
            print(f"[{datetime.now():%H:%M:%S}] Report generated: {output_path}")
            observer = Observer()
            handler = _Handler(lambda: _regenerate(data_path, output_path, compare_path, config, time_range))
            observer.schedule(handler, os.path.dirname(os.path.abspath(data_path)) or ".", recursive=False)
            observer.start()
            try:
                while True:
                    time.sleep(interval)
            finally:
                observer.stop()
                observer.join()

        run_with_watchdog()
    except ImportError:
        _run_polling(data_path, output_path, compare_path, config, time_range, interval)


def _regenerate(data_path, output_path, compare_path, config, time_range):
    generate_report(data_path, output_path, compare=compare_path, config=config, time_range=time_range)
    print(f"[{datetime.now():%H:%M:%S}] Report updated: {output_path}")


def _run_polling(data_path, output_path, compare_path, config, time_range, interval):
    generate_report(data_path, output_path, compare=compare_path, config=config, time_range=time_range)
    print(f"[{datetime.now():%H:%M:%S}] Report generated: {output_path}")
    last_data = _current_mtime(data_path)
    last_compare = _current_mtime(compare_path) if compare_path else None
    while True:
        time.sleep(interval)
        if _file_changed(data_path, last_data) or (compare_path and _file_changed(compare_path, last_compare)):
            _regenerate(data_path, output_path, compare_path, config, time_range)
            last_data = _current_mtime(data_path)
            last_compare = _current_mtime(compare_path) if compare_path else None
