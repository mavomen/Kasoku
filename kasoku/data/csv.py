"""CSV data source adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kasoku.data.base import (
    SERIES_ID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
    DataError,
    DataSource,
    validate_series,
)


class CsvSource(DataSource):
    """Read a time series from a CSV file.

    Column names are configurable. If ``series_id_column`` is not given, a
    column literally named ``symbol_or_series_id`` is used when present;
    otherwise a single-series id is derived from the file stem.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_column: str = TIMESTAMP_COL,
        value_column: str = VALUE_COL,
        series_id_column: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.timestamp_column = timestamp_column
        self.value_column = value_column
        self.series_id_column = series_id_column

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataError(f"csv file not found: {self.path}")
        try:
            raw = pd.read_csv(self.path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataError(f"failed to parse csv {self.path}: {exc}") from exc
        missing = [
            col
            for col in (self.timestamp_column, self.value_column)
            if col not in raw.columns
        ]
        if missing:
            raise DataError(f"csv {self.path} is missing required columns: {missing}")
        if raw.empty:
            raise DataError(f"csv {self.path} contains no data rows")
        frame = pd.DataFrame(
            {
                TIMESTAMP_COL: self._parse_timestamps(raw[self.timestamp_column]),
                SERIES_ID_COL: self._resolve_series_id(raw),
                VALUE_COL: self._parse_values(raw[self.value_column]),
            }
        )
        validate_series(frame)
        return frame

    def _parse_timestamps(self, column: pd.Series) -> pd.Series:
        try:
            timestamps = pd.to_datetime(column)
        except (ValueError, TypeError) as exc:
            raise DataError(
                f"csv {self.path}: timestamp column {self.timestamp_column!r} "
                f"contains unparseable values"
            ) from exc
        if timestamps.isna().any():
            raise DataError(
                f"csv {self.path}: timestamp column {self.timestamp_column!r} "
                f"contains unparseable values"
            )
        return timestamps

    def _parse_values(self, column: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(column)
        except (ValueError, TypeError) as exc:
            raise DataError(
                f"csv {self.path}: value column {self.value_column!r} "
                f"must be numeric"
            ) from exc

    def _resolve_series_id(self, raw: pd.DataFrame) -> pd.Series:
        column = self.series_id_column
        if column is None and SERIES_ID_COL in raw.columns:
            column = SERIES_ID_COL
        if column is not None:
            if column not in raw.columns:
                raise DataError(
                    f"csv {self.path} has no series id column {column!r}"
                )
            series = raw[column]
            if series.isna().any():
                raise DataError(
                    f"csv {self.path}: series id column {column!r} "
                    f"contains missing values"
                )
            return series.astype(str)
        stem = self.path.stem
        return pd.Series([stem] * len(raw), index=raw.index)
