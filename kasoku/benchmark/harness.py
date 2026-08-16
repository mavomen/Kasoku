"""Benchmark measurement primitives.

These helpers measure any callable (a compiled model inference, or later a
train step) and are deliberately generic: the caller decides what to measure
and passes the callable plus its call arguments. Nothing here wraps a callable
except :func:`compile_overhead`, which must observe the very first call so that
``tf.function``/XLA compilation is captured and kept separate from steady-state
timing. Measure compile overhead before steady-state measurements on the same
freshly wrapped callable, and never average the two together.
"""

import resource
import threading
from enum import Enum
from time import perf_counter
from typing import Callable

from kasoku.benchmark.base import BenchmarkError
from kasoku.execution import ExecutionMode, compile_call, resolve_mode

__all__ = [
    "BenchmarkError",
    "Phase",
    "compile_overhead",
    "peak_memory_mb",
    "resolve_phase",
    "throughput",
    "time_phase",
]


class Phase(str, Enum):
    """Benchmark phases. String values match the config ``VALID_PHASES``."""

    TRAIN = "train"
    INFERENCE = "inference"


VALID_PHASES = frozenset(phase.value for phase in Phase)


def resolve_phase(name: str | Phase) -> Phase:
    """Resolve a phase name (or ``Phase``) into a ``Phase``.

    Raises :class:`BenchmarkError` when ``name`` is not a valid phase, listing
    the accepted values.
    """
    if isinstance(name, Phase):
        return name
    try:
        return Phase(name)
    except ValueError as exc:
        raise BenchmarkError(
            f"unknown benchmark phase {name!r}; expected one of {sorted(VALID_PHASES)}"
        ) from exc


def _validate_timing(iterations: int, warmup_iterations: int) -> None:
    if iterations < 1:
        raise BenchmarkError(f"iterations must be >= 1, got {iterations}")
    if warmup_iterations < 0:
        raise BenchmarkError(f"warmup_iterations must be >= 0, got {warmup_iterations}")


def time_phase(
    fn: Callable,
    *args,
    iterations: int = 3,
    warmup_iterations: int = 1,
) -> float:
    """Return the mean steady-state wall-clock seconds per call of ``fn(*args)``.

    Warmup calls are run but excluded from timing so the measured interval
    reflects steady-state performance, not first-call compilation or lazy init.
    """
    _validate_timing(iterations, warmup_iterations)
    for _ in range(warmup_iterations):
        fn(*args)
    start = perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = perf_counter() - start
    return elapsed / iterations


def throughput(samples_per_call: int, mean_wall_clock_seconds: float) -> float:
    """Return samples processed per second for a steady-state call duration.

    ``samples_per_call`` is the batch size of one invocation and
    ``mean_wall_clock_seconds`` comes from :func:`time_phase`.
    """
    if samples_per_call < 1:
        raise BenchmarkError(f"samples_per_call must be >= 1, got {samples_per_call}")
    return samples_per_call / mean_wall_clock_seconds


def compile_overhead(
    fn: Callable,
    *args,
    mode: str | ExecutionMode,
    iterations: int = 3,
    warmup_iterations: int = 1,
) -> float | None:
    """Return first-call compile overhead in seconds, or ``None`` for eager mode.

    ``fn`` must be the raw (unwrapped) callable: it is compiled for the
    requested mode here so the first call captures ``tf.function``/XLA
    compilation. Overhead is the first call's duration minus the steady-state
    mean, reported separately so it is never averaged into throughput.
    """
    if resolve_mode(mode) is ExecutionMode.EAGER:
        return None
    _validate_timing(iterations, warmup_iterations)
    wrapped = compile_call(fn, mode)
    start = perf_counter()
    wrapped(*args)
    first_call_seconds = perf_counter() - start
    steady_seconds = time_phase(
        wrapped,
        *args,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    return max(0.0, first_call_seconds - steady_seconds)


def _current_rss_bytes() -> float:
    """Return current process resident memory in bytes.

    Reads ``/proc/self/status`` VmRSS on Linux; falls back to the process
    high-water mark (``ru_maxrss``) where ``/proc`` is unavailable.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _MemoryProbe:
    """Background sampler that tracks peak resident memory while a phase runs."""

    def __init__(self, sampler: Callable[[], float] = _current_rss_bytes, interval_seconds: float = 0.005):
        self._sampler = sampler
        self._interval_seconds = interval_seconds
        self._peak_bytes = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._peak_bytes = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_bytes = max(self._peak_bytes, self._sampler())
            self._stop.wait(self._interval_seconds)

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return self._peak_bytes


def peak_memory_mb(
    fn: Callable,
    *args,
    iterations: int = 3,
    warmup_iterations: int = 1,
    sampler: Callable[[], float] = _current_rss_bytes,
) -> float:
    """Return the peak resident memory (MB) observed while running the phase.

    A background thread samples the supplied ``sampler`` (default: current RSS)
    throughout the warmup and timed calls and reports the maximum, converted to
    mebibytes.
    """
    _validate_timing(iterations, warmup_iterations)
    probe = _MemoryProbe(sampler)
    probe.start()
    try:
        for _ in range(warmup_iterations + iterations):
            fn(*args)
    finally:
        peak_bytes = probe.stop()
    return peak_bytes / (1024 * 1024)
