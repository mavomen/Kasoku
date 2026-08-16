from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from kasoku.config import VALID_MODES
from kasoku.data import CsvSource
from kasoku.execution import (
    ExecutionError,
    ExecutionMode,
    VALID_MODES as EXEC_VALID_MODES,
    compile_call,
    eager_call,
    graph_call,
    resolve_mode,
    xla_call,
)
from kasoku.models import build_gru_by_size
from kasoku.pipeline import window_series

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _windowed_fixture():
    df = CsvSource(FIXTURES / "series_single.csv").read()
    x, _ = window_series(df, lookback=24, horizon=1)
    return x


def _sine(length=24, n=3):
    t = np.arange(n * length) % length
    return (np.sin(2 * np.pi * t / length) + 1).reshape(n, length, 1).astype(np.float32)


def test_valid_modes_match_config():
    assert EXEC_VALID_MODES == VALID_MODES
    assert sorted(mode.value for mode in ExecutionMode) == sorted(VALID_MODES)


@pytest.mark.parametrize("name", ["eager", "tf_function", "xla"])
def test_resolve_mode_from_string(name):
    assert resolve_mode(name) is ExecutionMode(name)


@pytest.mark.parametrize("mode", list(ExecutionMode))
def test_resolve_mode_accepts_enum(mode):
    assert resolve_mode(mode) is mode


@pytest.mark.parametrize("bad", ["tpu", "auto", ""])
def test_resolve_mode_unknown_raises(bad):
    with pytest.raises(ExecutionError, match="unknown execution mode") as excinfo:
        resolve_mode(bad)
    for valid in sorted(VALID_MODES):
        assert valid in str(excinfo.value)


def test_eager_call_is_identity():
    def fn(x):
        return x

    assert eager_call(fn) is fn


def test_graph_call_wraps_with_tf_function():
    wrapped = graph_call(lambda x: x * 2)
    assert wrapped is not None
    assert wrapped(_sine()) is not None


def test_graph_call_jit_compile_flag_sets_xla():
    wrapped = graph_call(lambda x: x * 2, jit_compile=True)
    out = wrapped(_sine()).numpy()
    np.testing.assert_allclose(out, _sine() * 2, atol=1e-6)


def test_xla_call_runs_compiled():
    wrapped = xla_call(lambda x: x + 1)
    np.testing.assert_allclose(wrapped(_sine()).numpy(), _sine() + 1, atol=1e-6)


def test_compile_call_dispatch_modes():
    eager = compile_call(lambda x: x, "eager")
    graph = compile_call(lambda x: x, "tf_function")
    xla = compile_call(lambda x: x, "xla")
    assert not isinstance(eager, tf.types.experimental.PolymorphicFunction)
    assert isinstance(graph, tf.types.experimental.PolymorphicFunction)
    assert isinstance(xla, tf.types.experimental.PolymorphicFunction)


def test_compile_call_accepts_enum():
    wrapped = compile_call(lambda x: x, ExecutionMode.XLA)
    assert isinstance(wrapped, tf.types.experimental.PolymorphicFunction)


def test_compile_call_unknown_mode_raises():
    with pytest.raises(ExecutionError, match="unknown execution mode"):
        compile_call(lambda x: x, "quantum")


@pytest.mark.parametrize("size", ["small", "medium"])
def test_gru_equivalent_across_modes_fixture(size):
    model = build_gru_by_size(size, lookback=24, horizon=1, seed=42)
    x = _windowed_fixture()
    eager_out = compile_call(model, "eager")(x).numpy()
    graph_out = compile_call(model, "tf_function")(x).numpy()
    xla_out = compile_call(model, "xla")(x).numpy()
    assert eager_out.shape == (x.shape[0], 1)
    np.testing.assert_allclose(eager_out, graph_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eager_out, xla_out, rtol=1e-5, atol=1e-5)
