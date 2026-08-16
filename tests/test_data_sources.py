"""Unit tests for the data source adapters."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kasoku.data import (
    SERIES_ID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
    CsvSource,
    DataError,
    InfluxDBSource,
    PandasSource,
    validate_series,
)
from kasoku.data.influxdb import _records_to_frame, _resolve_series_id

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"
MULTI = FIXTURES / "series_multi.csv"


# ---------------------------------------------------------------- CSV


def test_csv_single_series_normalized():
    df = CsvSource(SINGLE).read()
    assert list(df.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]
    assert df[df[SERIES_ID_COL] == "series_single"].shape == (96, 3)
    assert pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL])
    assert pd.api.types.is_numeric_dtype(df[VALUE_COL])
    validate_series(df)


def test_csv_single_series_default_id_is_file_stem():
    df = CsvSource(SINGLE).read()
    assert set(df[SERIES_ID_COL]) == {"series_single"}


def test_csv_multi_series():
    df = CsvSource(MULTI).read()
    assert set(df[SERIES_ID_COL]) == {"A", "B"}
    assert df.shape == (192, 3)


def test_csv_custom_column_names(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text(
        "ts,val\n2024-01-01T00:00:00Z,1.5\n2024-01-01T01:00:00Z,2.5\n",
        encoding="utf-8",
    )
    df = CsvSource(path, timestamp_column="ts", value_column="val").read()
    assert list(df.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]
    assert df[SERIES_ID_COL].tolist() == ["custom", "custom"]
    assert df[VALUE_COL].tolist() == [1.5, 2.5]


def test_csv_explicit_series_id_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "t,v,s\n2024-01-01T00:00:00Z,1.5,X\n2024-01-01T00:00:00Z,2.5,Y\n",
        encoding="utf-8",
    )
    df = CsvSource(path, timestamp_column="t", value_column="v", series_id_column="s").read()
    assert set(df[SERIES_ID_COL]) == {"X", "Y"}


def test_csv_series_id_column_requested_but_absent(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("t,v\n2024-01-01T00:00:00Z,1.5\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path, timestamp_column="t", value_column="v", series_id_column="s").read()


def test_csv_missing_file(tmp_path):
    with pytest.raises(DataError):
        CsvSource(tmp_path / "nope.csv").read()


def test_csv_parser_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('a,b\n"1,2\n3\n', encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path, timestamp_column="a", value_column="b").read()


def test_csv_missing_required_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("timestamp\n2024-01-01T00:00:00Z\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path).read()


def test_csv_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path).read()


def test_csv_header_only(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("timestamp,value\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path).read()


def test_csv_unparseable_timestamp(tmp_path):
    path = tmp_path / "bad_ts.csv"
    path.write_text("timestamp,value\n2024-99-99,1.5\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path).read()


def test_csv_missing_timestamp_value(tmp_path):
    path = tmp_path / "missing_ts.csv"
    path.write_text("timestamp,value\n,1.5\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path).read()


def test_csv_non_numeric_value(tmp_path):
    path = tmp_path / "bad_val.csv"
    path.write_text("timestamp,value\n2024-01-01T00:00:00Z,abc\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvSource(path).read()


def test_csv_missing_series_id_values(tmp_path):
    path = tmp_path / "null_id.csv"
    path.write_text(
        "timestamp,symbol_or_series_id,value\n"
        "2024-01-01T00:00:00Z,A,1.5\n"
        "2024-01-01T01:00:00Z,,2.5\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError):
        CsvSource(path).read()


# ------------------------------------------------------------- Pandas


def test_pandas_canonical_roundtrip():
    frame = pd.DataFrame(
        {
            TIMESTAMP_COL: pd.to_datetime(["2024-01-01T00:00:00Z"] * 2),
            SERIES_ID_COL: ["A", "B"],
            VALUE_COL: [1.5, 2.5],
        }
    )
    df = PandasSource(frame).read()
    pd.testing.assert_frame_equal(df, frame)


def test_pandas_datetime_index_single_column():
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    frame = pd.DataFrame({"temp": [1.0, 2.0, 3.0]}, index=index)
    df = PandasSource(frame).read()
    assert list(df.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]
    assert df[SERIES_ID_COL].tolist() == ["default", "default", "default"]
    assert df[VALUE_COL].tolist() == [1.0, 2.0, 3.0]
    assert df[TIMESTAMP_COL].tolist() == index.tolist()


def test_pandas_datetime_index_custom_series_id():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    frame = pd.DataFrame({"temp": [1.0, 2.0]}, index=index)
    df = PandasSource(frame, series_id="room-1").read()
    assert df[SERIES_ID_COL].tolist() == ["room-1", "room-1"]


def test_pandas_missing_columns():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(DataError):
        PandasSource(frame).read()


def test_pandas_wrong_timestamp_dtype():
    frame = pd.DataFrame(
        {
            TIMESTAMP_COL: ["2024-01-01", "2024-01-02"],
            SERIES_ID_COL: ["A", "A"],
            VALUE_COL: [1.0, 2.0],
        }
    )
    with pytest.raises(DataError):
        PandasSource(frame).read()


def test_pandas_non_numeric_value():
    frame = pd.DataFrame(
        {
            TIMESTAMP_COL: pd.to_datetime(["2024-01-01", "2024-01-02"]),
            SERIES_ID_COL: ["A", "A"],
            VALUE_COL: ["x", "y"],
        }
    )
    with pytest.raises(DataError):
        PandasSource(frame).read()


def test_pandas_non_string_series_id():
    frame = pd.DataFrame(
        {
            TIMESTAMP_COL: pd.to_datetime(["2024-01-01", "2024-01-02"]),
            SERIES_ID_COL: [1, 2],
            VALUE_COL: [1.0, 2.0],
        }
    )
    with pytest.raises(DataError):
        PandasSource(frame).read()


def test_pandas_empty_canonical():
    frame = pd.DataFrame(columns=[TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL])
    with pytest.raises(DataError):
        PandasSource(frame).read()


def test_pandas_datetime_index_multiple_columns():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=index)
    with pytest.raises(DataError):
        PandasSource(frame).read()


def test_pandas_datetime_index_non_numeric_column():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    frame = pd.DataFrame({"a": ["x", "y"]}, index=index)
    with pytest.raises(DataError):
        PandasSource(frame).read()


# ------------------------------------------------------------ InfluxDB


class _FakeRecord:
    def __init__(self, time, value, measurement="meas", field="value", tags=None):
        self._time = time
        self._value = value
        self.values = {"_measurement": measurement, "_field": field, **(tags or {})}

    def get_time(self):
        return self._time

    def get_value(self):
        return self._value


class _FakeQueryAPI:
    def __init__(self, res):
        self.res = res
        self.last_query = None
        self.last_org = None

    def query(self, query, org=None):
        self.last_query = query
        self.last_org = org
        return self.res


class _FakeInfluxDBClient:
    instances: list["_FakeInfluxDBClient"] = []

    def __init__(self, url=None, token=None, org=None, **kwargs):
        self.url = url
        self.token = token
        self.org = org
        self.api = _FakeQueryAPI(_FakeInfluxDBClient.result)
        self.closed = False
        _FakeInfluxDBClient.instances.append(self)

    def query_api(self):
        return self.api

    def close(self):
        self.closed = True


def _install_fake_client(monkeypatch, result):
    _FakeInfluxDBClient.result = result
    _FakeInfluxDBClient.instances = []
    monkeypatch.setitem(
        sys.modules,
        "influxdb_client",
        types.SimpleNamespace(InfluxDBClient=_FakeInfluxDBClient),
    )
    return _FakeInfluxDBClient


def test_influx_missing_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "influxdb_client", None)
    source = InfluxDBSource("http://localhost:8086", "t", "o", "b")
    with pytest.raises(DataError, match="influxdb-client"):
        source.read()


def test_influx_read_with_fake_client(monkeypatch):
    result = [
        [
            _FakeRecord(pd.Timestamp("2024-01-01T00:00:00Z"), 1.5),
            _FakeRecord(pd.Timestamp("2024-01-01T01:00:00Z"), 2.5),
        ],
        [_FakeRecord(pd.Timestamp("2024-01-01T00:00:00Z"), 3.5, measurement="m2", field="f2")],
    ]
    fake = _install_fake_client(monkeypatch, result)
    source = InfluxDBSource("http://localhost:8086", "t", "org1", "bucket1")
    df = source.read()

    assert list(df.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]
    assert df[SERIES_ID_COL].tolist() == ["meas.value", "meas.value", "m2.f2"]
    assert df[VALUE_COL].tolist() == [1.5, 2.5, 3.5]
    assert fake.instances[0].closed


def test_influx_default_query_from_bucket(monkeypatch):
    _install_fake_client(monkeypatch, [[_FakeRecord(pd.Timestamp("2024-01-01"), 1.0)]])
    source = InfluxDBSource("http://localhost:8086", "t", "org1", "telemetry")
    source.read()
    client = _FakeInfluxDBClient.instances[-1]
    assert client.api.last_query == 'from(bucket: "telemetry") |> range(start: -1h)'
    assert client.api.last_org == "org1"


def test_influx_custom_query(monkeypatch):
    _install_fake_client(monkeypatch, [[_FakeRecord(pd.Timestamp("2024-01-01"), 1.0)]])
    source = InfluxDBSource("http://localhost:8086", "t", "org1", "b", 'from(bucket: "x")')
    source.read()
    assert _FakeInfluxDBClient.instances[-1].api.last_query == 'from(bucket: "x")'


def test_influx_series_id_tag(monkeypatch):
    result = [
        [
            _FakeRecord(
                pd.Timestamp("2024-01-01T00:00:00Z"),
                1.5,
                tags={"host": "h1"},
            )
        ]
    ]
    _install_fake_client(monkeypatch, result)
    source = InfluxDBSource("http://localhost:8086", "t", "org1", "b", series_id_tag="host")
    df = source.read()
    assert df[SERIES_ID_COL].tolist() == ["h1"]


def test_influx_empty_result(monkeypatch):
    _install_fake_client(monkeypatch, [])
    source = InfluxDBSource("http://localhost:8086", "t", "org1", "b")
    with pytest.raises(DataError, match="no rows"):
        source.read()


def test_influx_client_connection_kwargs(monkeypatch):
    _install_fake_client(monkeypatch, [[_FakeRecord(pd.Timestamp("2024-01-01"), 1.0)]])
    source = InfluxDBSource("http://localhost:9999", "tok", "org1", "b")
    source.read()
    client = _FakeInfluxDBClient.instances[-1]
    assert client.url == "http://localhost:9999"
    assert client.token == "tok"
    assert client.org == "org1"
    assert client.closed


@pytest.mark.parametrize(
    ("values", "tag", "expected"),
    [
        ({"host": "h1", "_field": "v", "_measurement": "m"}, "host", "h1"),
        ({"host": None, "_field": "v", "_measurement": "m"}, "host", "m.v"),
        ({"_field": "v", "_measurement": "m"}, None, "m.v"),
        ({"_measurement": "m"}, None, "m"),
        ({"_field": "v"}, None, "v"),
        ({}, None, "default"),
        ({"host": float("nan"), "_measurement": "m"}, "host", "m"),
    ],
)
def test_influx_resolve_series_id(values, tag, expected):
    assert (
        _resolve_series_id(
            values,
            series_id_tag=tag,
            field=values.get("_field"),
            measurement=values.get("_measurement"),
        )
        == expected
    )


def test_influx_records_to_frame():
    rows = [
        {TIMESTAMP_COL: pd.Timestamp("2024-01-01T00:00:00Z"), SERIES_ID_COL: "A", VALUE_COL: 1.0},
        {TIMESTAMP_COL: pd.Timestamp("2024-01-01T01:00:00Z"), SERIES_ID_COL: "A", VALUE_COL: np.float64(2.0)},
    ]
    df = _records_to_frame(rows)
    assert pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL])
    assert df[VALUE_COL].tolist() == [1.0, 2.0]


def test_validate_series_empty_dataframe():
    with pytest.raises(DataError):
        validate_series(pd.DataFrame())
