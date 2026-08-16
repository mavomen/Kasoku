"""Chronological train/val/test splitting.

Each series is split independently by row fraction (``train``, ``val``,
``test``) after sorting by timestamp, so temporal order is preserved and
series of different lengths each contribute to every split.
"""

from __future__ import annotations

import math

import pandas as pd

from kasoku.data import SERIES_ID_COL, TIMESTAMP_COL
from kasoku.pipeline.base import PipelineError, validate_canonical

_SPLIT_RATIOS = ("train", "val", "test")


def train_val_test_split(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a canonical series frame into train, val, and test frames.

    Row counts per series follow ``floor(ratio * n)`` for train and val,
    with the remainder going to test. A series shorter than a split's
    allocation simply contributes nothing to that split.
    """
    validate_canonical(df)
    _validate_ratios(train, val, test)

    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for _, series in df.groupby(SERIES_ID_COL, sort=False):
        series = series.sort_values(TIMESTAMP_COL, kind="stable")
        n = len(series)
        train_n = int(train * n)
        val_n = int(val * n)
        train_frames.append(series.iloc[:train_n])
        val_frames.append(series.iloc[train_n : train_n + val_n])
        test_frames.append(series.iloc[train_n + val_n :])
    return (
        pd.concat(train_frames),
        pd.concat(val_frames),
        pd.concat(test_frames),
    )


def _validate_ratios(train: float, val: float, test: float) -> None:
    ratios = (train, val, test)
    for name, ratio in zip(_SPLIT_RATIOS, ratios):
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise PipelineError(f"{name} ratio must be a number, got {type(ratio).__name__}")
        if not 0.0 < ratio < 1.0:
            raise PipelineError(f"{name} ratio must be in (0, 1), got {ratio}")
    total = train + val + test
    if not math.isclose(total, 1.0, rel_tol=1e-9):
        raise PipelineError(f"train + val + test must sum to 1, got {total}")
