"""Data source interface and canonical series schema.

Every data source (CSV, InfluxDB, generic pandas-compatible) normalizes to
the same long-form DataFrame:

- ``timestamp``: datetime (timezone-aware or naive)
- ``symbol_or_series_id``: str, the series identifier
- ``value``: float, the observed value
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

TIMESTAMP_COL = "timestamp"
SERIES_ID_COL = "symbol_or_series_id"
VALUE_COL = "value"

CANONICAL_COLUMNS = (TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL)


class DataError(ValueError):
    """Raised when a data source cannot be read or normalized."""


class DataSource(ABC):
    """Base class for all data sources."""

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read the source into a canonical long-form DataFrame."""


def validate_series(df: pd.DataFrame) -> None:
    """Validate that *df* conforms to the canonical series schema.

    Raises :class:`DataError` if required columns are missing, dtypes are
    wrong, or the frame is empty.
    """
    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise DataError(f"missing required columns: {missing}")
    if df.empty:
        raise DataError("series is empty")
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
        raise DataError("timestamp column must be datetime64")
    if not pd.api.types.is_numeric_dtype(df[VALUE_COL]):
        raise DataError("value column must be numeric")
    if not df[SERIES_ID_COL].map(lambda value: isinstance(value, str)).all():
        raise DataError(f"{SERIES_ID_COL} must contain only strings")
