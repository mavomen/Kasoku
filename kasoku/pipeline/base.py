"""Shared pipeline errors and validation helpers."""

from __future__ import annotations

import pandas as pd

from kasoku.data import validate_series


class PipelineError(ValueError):
    """Raised when a pipeline step fails or receives invalid input."""


def validate_canonical(df: pd.DataFrame) -> None:
    """Validate *df* against the canonical series schema.

    Raises :class:`PipelineError` (wrapping :class:`~kasoku.data.DataError`)
    when the frame is not a valid canonical series.
    """
    try:
        validate_series(df)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
