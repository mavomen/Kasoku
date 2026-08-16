"""Z-score normalization with inverse-transform support.

Stats are always fit on a single reference frame (normally the training
split) and reused to transform other frames and to un-scale predictions.
By default every series is normalized against its own mean/std.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kasoku.data import SERIES_ID_COL, VALUE_COL
from kasoku.pipeline.base import PipelineError, validate_canonical

_GLOBAL_KEY = "__global__"


class Normalizer:
    """Per-series (or global) z-score scaler with inverse transform."""

    def __init__(self, *, per_series: bool = True) -> None:
        self.per_series = per_series
        self._params: dict[str, tuple[float, float]] = {}

    @property
    def params(self) -> dict[str, tuple[float, float]]:
        """Per-series (mean, std) pairs; useful for checkpointing."""
        return dict(self._params)

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """Compute normalization stats from *df* (typically training data)."""
        validate_canonical(df)
        if self.per_series:
            for series_id, series in df.groupby(SERIES_ID_COL, sort=False):
                values = _as_float64(series[VALUE_COL])
                if pd.isna(values).any():
                    raise PipelineError(f"series {series_id!r} contains missing values")
                self._params[str(series_id)] = (
                    float(values.mean()),
                    float(values.std(ddof=0)),
                )
        else:
            values = _as_float64(df[VALUE_COL])
            self._params[_GLOBAL_KEY] = (
                float(values.mean()),
                float(values.std(ddof=0)),
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale *df* in place of a copy, returning a new frame."""
        validate_canonical(df)
        out = df.copy()
        if self.per_series:
            for series_id in out[SERIES_ID_COL].unique():
                mask = out[SERIES_ID_COL] == series_id
                out.loc[mask, VALUE_COL] = self.scale(out.loc[mask, VALUE_COL], series_id)
        else:
            out[VALUE_COL] = self.scale(out[VALUE_COL], None)
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore *df* to its original scale, returning a new frame."""
        validate_canonical(df)
        out = df.copy()
        if self.per_series:
            for series_id in out[SERIES_ID_COL].unique():
                mask = out[SERIES_ID_COL] == series_id
                out.loc[mask, VALUE_COL] = self.unscale(out.loc[mask, VALUE_COL], series_id)
        else:
            out[VALUE_COL] = self.unscale(out[VALUE_COL], None)
        return out

    def scale(self, values: np.ndarray, series_id: str | None) -> np.ndarray:
        """Standardize raw values for a series (float64 in, float64 out)."""
        mean, std = self._stats_for(series_id)
        arr = _as_float64(values)
        if std == 0.0:
            return np.zeros_like(arr)
        return (arr - mean) / std

    def unscale(self, values: np.ndarray, series_id: str | None) -> np.ndarray:
        """Undo standardization for a series (float64 in, float64 out)."""
        mean, std = self._stats_for(series_id)
        arr = _as_float64(values)
        return arr * std + mean

    def _stats_for(self, series_id: str | None) -> tuple[float, float]:
        key = str(series_id) if self.per_series else _GLOBAL_KEY
        try:
            return self._params[key]
        except KeyError as exc:
            raise PipelineError(
                f"series {series_id!r} was not present during fit"
            ) from exc


def _as_float64(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)
