"""Windowing: convert canonical series into supervised (X, y) windows.

Windows are produced per series (in appearance order) and pooled into two
contiguous float32 arrays:

- ``X`` with shape ``(n_samples, lookback, 1)`` -- the ``lookback`` most
  recent observations for each sample
- ``y`` with shape ``(n_samples, horizon)`` -- the next ``horizon``
  observations

The trailing single feature axis keeps the output directly consumable by
the model layer without a reshape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kasoku.data import SERIES_ID_COL, TIMESTAMP_COL, VALUE_COL
from kasoku.pipeline.base import PipelineError, validate_canonical


def window_series(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised windows from a canonical long-form series frame."""
    validate_canonical(df)
    if lookback < 1:
        raise PipelineError(f"lookback must be >= 1, got {lookback}")
    if horizon < 1:
        raise PipelineError(f"horizon must be >= 1, got {horizon}")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for series_id, series in df.groupby(SERIES_ID_COL, sort=False):
        series = series.sort_values(TIMESTAMP_COL, kind="stable")
        if series[TIMESTAMP_COL].duplicated().any():
            raise PipelineError(f"series {series_id!r} contains duplicate timestamps")
        if series[VALUE_COL].isna().any():
            raise PipelineError(f"series {series_id!r} contains missing values")
        values = series[VALUE_COL].to_numpy(dtype=np.float32)
        count = len(values) - lookback - horizon + 1
        if count < 1:
            continue
        xs = np.stack([values[i : i + lookback] for i in range(count)])
        ys = np.stack([values[i + lookback : i + lookback + horizon] for i in range(count)])
        all_x.append(xs[..., None])
        all_y.append(ys)

    if not all_x:
        raise PipelineError(
            "no windows could be produced: every series is shorter than "
            f"lookback + horizon ({lookback + horizon})"
        )
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
