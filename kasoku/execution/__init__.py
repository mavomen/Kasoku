"""Execution-mode wrappers: eager, tf.function, and XLA.

A model (or any callable) is wrapped exactly once and run in the requested
mode. The mode is a runtime choice selected via :func:`compile_call`; there is
never a separate code path or duplicated model class per mode.
"""

from kasoku.execution.base import ExecutionError
from kasoku.execution.modes import (
    VALID_MODES,
    ExecutionMode,
    compile_call,
    eager_call,
    graph_call,
    resolve_mode,
    xla_call,
)

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
