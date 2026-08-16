import threading
import time

import pytest

from kasoku.benchmark import (
    BenchmarkError,
    Phase,
    compile_overhead,
    peak_memory_mb,
    resolve_phase,
    throughput,
    time_phase,
)
from kasoku.benchmark.harness import _current_rss_bytes
from kasoku.config import VALID_PHASES


def test_phase_values_match_config():
    assert sorted(phase.value for phase in Phase) == sorted(VALID_PHASES)


@pytest.mark.parametrize("name", ["train", "inference"])
def test_resolve_phase_from_string(name):
    assert resolve_phase(name) is Phase(name)


@pytest.mark.parametrize("phase", list(Phase))
def test_resolve_phase_accepts_enum(phase):
    assert resolve_phase(phase) is phase


@pytest.mark.parametrize("bad", ["validation", "eval", ""])
def test_resolve_phase_unknown_raises(bad):
    with pytest.raises(BenchmarkError, match="unknown benchmark phase") as excinfo:
        resolve_phase(bad)
    for valid in sorted(VALID_PHASES):
        assert valid in str(excinfo.value)


def _sleepy(seconds=0.005):
    def fn(_):
        time.sleep(seconds)
        return _

    return fn


def test_time_phase_measures_steady_state():
    mean = time_phase(_sleepy(0.005), 1, iterations=3, warmup_iterations=1)
    assert 0.003 < mean < 0.05


def test_time_phase_excludes_warmup():
    calls = []
    start = time.perf_counter()

    def slow_first_then_fast(x):
        if not calls:
            time.sleep(0.05)
        else:
            time.sleep(0.002)
        calls.append(x)
        return x

    mean = time_phase(slow_first_then_fast, 1, iterations=4, warmup_iterations=1)
    assert len(calls) == 5
    assert mean < 0.01


def test_time_phase_validates_iterations():
    with pytest.raises(BenchmarkError, match="iterations"):
        time_phase(_sleepy(), 1, iterations=0)
    with pytest.raises(BenchmarkError, match="warmup_iterations"):
        time_phase(_sleepy(), 1, warmup_iterations=-1)


def test_throughput_calculation():
    assert throughput(samples_per_call=32, mean_wall_clock_seconds=0.004) == pytest.approx(8000)


def test_throughput_validates_samples_per_call():
    with pytest.raises(BenchmarkError, match="samples_per_call"):
        throughput(samples_per_call=0, mean_wall_clock_seconds=0.004)


def _slow_first_call_factory():
    calls = []

    def fn(x):
        if not calls:
            time.sleep(0.05)
        else:
            time.sleep(0.002)
        calls.append(1)
        return x

    return fn


def test_compile_overhead_captures_first_call_for_graph():
    overhead = compile_overhead(_slow_first_call_factory(), 1, mode="tf_function", iterations=4)
    assert overhead is not None
    assert 0.01 < overhead < 2.0


def test_compile_overhead_captures_first_call_for_xla():
    overhead = compile_overhead(_slow_first_call_factory(), 1, mode="xla", iterations=4)
    assert overhead is not None
    assert 0.01 < overhead < 2.0


def test_compile_overhead_is_none_for_eager():
    overhead = compile_overhead(_slow_first_call_factory(), 1, mode="eager", iterations=4)
    assert overhead is None


def test_compile_overhead_validates_iterations():
    with pytest.raises(BenchmarkError, match="iterations"):
        compile_overhead(_sleepy(), 1, mode="xla", iterations=0)


def test_peak_memory_mb_uses_sampler_peak():
    samples = iter([0.0, 1024 * 1024, 512 * 1024])

    def sampler():
        return next(samples)

    peak = peak_memory_mb(_sleepy(0.002), 1, iterations=2, warmup_iterations=1, sampler=sampler)
    assert peak == pytest.approx(1.0)


def test_peak_memory_mb_runs_with_default_sampler():
    peak = peak_memory_mb(_sleepy(0.002), 1, iterations=2, warmup_iterations=1)
    assert peak >= 0.0


def test_peak_memory_mb_validates_iterations():
    with pytest.raises(BenchmarkError, match="iterations"):
        peak_memory_mb(_sleepy(), 1, iterations=0)


def test_current_rss_bytes_returns_positive():
    assert _current_rss_bytes() > 0.0


def test_current_rss_bytes_falls_back_when_proc_unavailable(monkeypatch):
    def raising_open(*args, **kwargs):
        raise OSError("no proc")

    monkeypatch.setattr("builtins.open", raising_open)
    assert _current_rss_bytes() > 0.0


def test_memory_probe_tracks_peak_and_releases_thread():
    from kasoku.benchmark.harness import _MemoryProbe

    seq = [0.0, 10.0, 50.0, 30.0, 5.0]
    index = [0]

    def sampler():
        value = seq[min(index[0], len(seq) - 1)]
        index[0] += 1
        return value

    probe = _MemoryProbe(sampler, interval_seconds=0.001)
    probe.start()
    time.sleep(0.05)
    assert probe.stop() == pytest.approx(50.0)
    assert probe._thread is None
    probe.start()
    index[0] = 0
    time.sleep(0.05)
    assert probe.stop() == pytest.approx(50.0)
