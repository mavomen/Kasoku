"""Generic pandas-compatible data source adapter."""

from __future__ import annotations

import pandas as pd

from kasoku.data.base import (
    CANONICAL_COLUMNS,
    SERIES_ID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
    DataError,
    DataSource,
    validate_series,
)


class PandasSource(DataSource):
    """Wrap an in-memory DataFrame as a data source.

    Accepts either a frame already in the canonical long-form schema or a
    single-series frame with a :class:`pd.DatetimeIndex` and exactly one
    numeric column.
    """

    def __init__(self, frame: pd.DataFrame, *, series_id: str = "default") -> None:
        self.frame = frame
        self.series_id = series_id

    def read(self) -> pd.DataFrame:
        frame = self.frame
        if set(CANONICAL_COLUMNS).issubset(frame.columns):
            validate_series(frame)
            return frame.copy()
        if isinstance(frame.index, pd.DatetimeIndex) and len(frame.columns) == 1:
            column = frame.columns[0]
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise DataError("value column must be numeric")
            normalized = pd.DataFrame(
                {
                    TIMESTAMP_COL: frame.index,
                    SERIES_ID_COL: self.series_id,
                    VALUE_COL: frame[column].to_numpy(),
                }
            )
            validate_series(normalized)
            return normalized
        raise DataError(
            "expected canonical columns (timestamp, symbol_or_series_id, value) "
            "or a DataFrame with a DatetimeIndex and a single numeric column"
        )
