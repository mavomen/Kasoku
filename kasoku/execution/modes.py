"""Execution-mode wrappers: eager, tf.function, and XLA.

A model or any callable is wrapped exactly once and run in the requested mode;
the mode is a runtime choice selected via :func:`compile_call`, never a
separate code path. ``eager_call`` is the identity wrapper, ``graph_call``
traces ``fn`` with ``tf.function``, and ``xla_call`` requests
``tf.function(jit_compile=True)``.
"""

import tensorflow as tf
from enum import Enum
from typing import Callable

from kasoku.execution.base import ExecutionError

__all__ = [
    "ExecutionError",
    "ExecutionMode",
    "VALID_MODES",
    "compile_call",
    "eager_call",
    "graph_call",
    "resolve_mode",
    "xla_call",
]


class ExecutionMode(str, Enum):
    """Named execution modes. String values match the config ``VALID_MODES``."""

    EAGER = "eager"
    TF_FUNCTION = "tf_function"
    XLA = "xla"


VALID_MODES = frozenset(mode.value for mode in ExecutionMode)


def eager_call(fn: Callable) -> Callable:
    """Return ``fn`` unchanged so it runs under eager execution."""
    return fn


def graph_call(fn: Callable, *, jit_compile: bool = False) -> Callable:
    """Trace ``fn`` with ``tf.function``, optionally requesting XLA compilation."""
    return tf.function(fn, jit_compile=jit_compile)


def xla_call(fn: Callable) -> Callable:
    """Trace ``fn`` with ``tf.function(jit_compile=True)`` so XLA compiles it."""
    return graph_call(fn, jit_compile=True)


def resolve_mode(name: str | ExecutionMode) -> ExecutionMode:
    """Resolve a mode name (or ``ExecutionMode``) into an ``ExecutionMode``.

    Raises :class:`ExecutionError` when ``name`` is not a valid mode, listing
    the accepted values.
    """
    if isinstance(name, ExecutionMode):
        return name
    try:
        return ExecutionMode(name)
    except ValueError as exc:
        raise ExecutionError(
            f"unknown execution mode {name!r}; expected one of {sorted(VALID_MODES)}"
        ) from exc


def compile_call(fn: Callable, mode: str | ExecutionMode) -> Callable:
    """Wrap ``fn`` for the requested execution mode.

    ``fn`` may be a model (for inference) or any other callable such as a train
    step. The same object is wrapped once; mode is purely a runtime choice.
    """
    resolved = resolve_mode(mode)
    if resolved is ExecutionMode.EAGER:
        return eager_call(fn)
    if resolved is ExecutionMode.TF_FUNCTION:
        return graph_call(fn)
    return xla_call(fn)
