"""InfluxDB (v2, FLUX) data source adapter.

The ``influxdb-client`` package is imported lazily so it is never a hard
dependency: the adapter only fails when actually used without the client
installed.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from kasoku.data.base import (
    SERIES_ID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
    DataError,
    DataSource,
    validate_series,
)


class InfluxDBSource(DataSource):
    """Query an InfluxDB 2.x bucket with a FLUX query.

    If ``query`` is not provided, a default query is built from ``bucket``.
    The series id is taken from ``series_id_tag`` when configured and
    present, otherwise from the measurement/field pair.
    """

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        query: str | None = None,
        *,
        series_id_tag: str | None = None,
    ) -> None:
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.query = query
        self.series_id_tag = series_id_tag

    def read(self) -> pd.DataFrame:
        try:
            from influxdb_client import InfluxDBClient
        except ImportError as exc:
            raise DataError(
                "influxdb-client is required for the influxdb source; "
                "install it with 'pip install influxdb-client'"
            ) from exc
        query = self.query or f'from(bucket: "{self.bucket}") |> range(start: -1h)'
        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        try:
            result = client.query_api().query(query=query, org=self.org)
            rows = [
                self._record_to_row(record)
                for table in result
                for record in table
            ]
        finally:
            client.close()
        if not rows:
            raise DataError(f"influxdb query returned no rows: {query}")
        frame = _records_to_frame(rows)
        validate_series(frame)
        return frame

    def _record_to_row(self, record: object) -> dict:
        values = record.values
        field = values.get("_field")
        measurement = values.get("_measurement")
        series_id = _resolve_series_id(
            values,
            series_id_tag=self.series_id_tag,
            field=field,
            measurement=measurement,
        )
        return {
            TIMESTAMP_COL: record.get_time(),
            SERIES_ID_COL: series_id,
            VALUE_COL: record.get_value(),
        }


def _resolve_series_id(
    values: Mapping,
    *,
    series_id_tag: str | None,
    field: object,
    measurement: object,
) -> str:
    if series_id_tag:
        tag = values.get(series_id_tag)
        if tag is not None and not pd.isna(tag):
            return str(tag)
    if measurement and field:
        return f"{measurement}.{field}"
    if measurement:
        return str(measurement)
    if field:
        return str(field)
    return "default"


def _records_to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL])
    frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL])
    return frame
