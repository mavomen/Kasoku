"""Unit tests for the data pipeline: windowing, split, normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kasoku.data import SERIES_ID_COL, TIMESTAMP_COL, VALUE_COL, CsvSource
from kasoku.pipeline import (
    Normalizer,
    PipelineError,
    train_val_test_split,
    window_series,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
SINGLE = FIXTURES / "series_single.csv"
MULTI = FIXTURES / "series_multi.csv"


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL])


def _timestamped(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    return _frame(
        [
            (ts, "A", value)
            for ts, value in zip(pd.date_range(start, periods=len(values), freq="h"), values)
        ]
    )


# ------------------------------------------------------------- windowing


def test_window_single_shapes():
    df = CsvSource(SINGLE).read()
    x, y = window_series(df, lookback=24, horizon=1)
    assert x.shape == (72, 24, 1)
    assert y.shape == (72, 1)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_window_multi_pooled():
    df = CsvSource(MULTI).read()
    x, y = window_series(df, lookback=24, horizon=1)
    assert x.shape == (144, 24, 1)
    assert y.shape == (144, 1)


def test_window_content_lookback_and_horizon():
    df = _timestamped([float(v) for v in range(10)])
    x, y = window_series(df, lookback=3, horizon=1)
    assert x.shape == (7, 3, 1)
    assert x[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert y[0].tolist() == [3.0]
    assert x[1, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert y[1].tolist() == [4.0]


def test_window_multi_step_horizon():
    df = _timestamped([float(v) for v in range(10)])
    x, y = window_series(df, lookback=3, horizon=2)
    assert y.shape == (6, 2)
    assert y[0].tolist() == [3.0, 4.0]
    assert y[5].tolist() == [8.0, 9.0]


@pytest.mark.parametrize("lookback,horizon", [(0, 1), (1, 0), (-3, 1), (1, -2)])
def test_window_non_positive_dims(lookback, horizon):
    df = _timestamped([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(PipelineError):
        window_series(df, lookback=lookback, horizon=horizon)


def test_window_empty_frame():
    df = pd.DataFrame(columns=[TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL])
    with pytest.raises(PipelineError):
        window_series(df, lookback=3, horizon=1)


def test_window_non_canonical_frame():
    with pytest.raises(PipelineError):
        window_series(pd.DataFrame({"a": [1, 2]}), lookback=1, horizon=1)


def test_window_duplicate_timestamps():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 1.0),
            (pd.Timestamp("2024-01-01"), "A", 2.0),
            (pd.Timestamp("2024-01-02"), "A", 3.0),
        ]
    )
    with pytest.raises(PipelineError, match="duplicate"):
        window_series(df, lookback=1, horizon=1)


def test_window_missing_values():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 1.0),
            (pd.Timestamp("2024-01-02"), "A", float("nan")),
            (pd.Timestamp("2024-01-03"), "A", 3.0),
        ]
    )
    with pytest.raises(PipelineError, match="missing"):
        window_series(df, lookback=1, horizon=1)


def test_window_series_too_short_for_dims():
    df = _timestamped([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(PipelineError, match="no windows"):
        window_series(df, lookback=4, horizon=1)


def test_window_skips_short_series_keeps_long():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 1.0),
            (pd.Timestamp("2024-01-02"), "A", 2.0),
            (pd.Timestamp("2024-01-03"), "A", 3.0),
            (pd.Timestamp("2024-01-04"), "A", 4.0),
            (pd.Timestamp("2024-01-05"), "A", 5.0),
            (pd.Timestamp("2024-01-06"), "A", 6.0),
            (pd.Timestamp("2024-01-01"), "B", 1.0),
            (pd.Timestamp("2024-01-02"), "B", 2.0),
        ]
    )
    x, y = window_series(df, lookback=3, horizon=1)
    assert x.shape == (3, 3, 1)
    assert y.shape == (3, 1)


def test_window_sorts_unsorted_input():
    df = _frame(
        [
            (pd.Timestamp("2024-01-04"), "A", 4.0),
            (pd.Timestamp("2024-01-01"), "A", 1.0),
            (pd.Timestamp("2024-01-03"), "A", 3.0),
            (pd.Timestamp("2024-01-02"), "A", 2.0),
            (pd.Timestamp("2024-01-05"), "A", 5.0),
        ]
    )
    x, y = window_series(df, lookback=2, horizon=1)
    assert x[0, :, 0].tolist() == [1.0, 2.0]
    assert y[0].tolist() == [3.0]
    assert x[2, :, 0].tolist() == [3.0, 4.0]
    assert y[2].tolist() == [5.0]


# --------------------------------------------------------------- split


def test_split_single_counts():
    df = CsvSource(SINGLE).read()
    train, val, test = train_val_test_split(df, 0.7, 0.15, 0.15)
    assert len(train) == 67
    assert len(val) == 14
    assert len(test) == 15
    assert len(train) + len(val) + len(test) == len(df)


def test_split_multi_per_series():
    df = CsvSource(MULTI).read()
    train, val, test = train_val_test_split(df, 0.7, 0.15, 0.15)
    for split in (train, val, test):
        assert set(split[SERIES_ID_COL].unique()) == {"A", "B"}
    assert len(train) == 134
    assert len(val) == 28
    assert len(test) == 30


def test_split_is_chronological():
    df = CsvSource(SINGLE).read()
    train, val, test = train_val_test_split(df, 0.7, 0.15, 0.15)
    assert train[TIMESTAMP_COL].max() <= val[TIMESTAMP_COL].min()
    assert val[TIMESTAMP_COL].max() <= test[TIMESTAMP_COL].min()
    assert train[TIMESTAMP_COL].max() < test[TIMESTAMP_COL].min()


def test_split_preserves_canonical_schema():
    df = CsvSource(MULTI).read()
    train, val, test = train_val_test_split(df, 0.7, 0.15, 0.15)
    assert list(train.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]
    assert list(val.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]
    assert list(test.columns) == [TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL]


def test_split_short_series_empty_val():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 1.0),
            (pd.Timestamp("2024-01-02"), "A", 2.0),
        ]
    )
    train, val, test = train_val_test_split(df, 0.5, 0.25, 0.25)
    assert len(train) == 1
    assert len(val) == 0
    assert len(test) == 1


@pytest.mark.parametrize("ratios", [(0.7, 0.15, 0.2), (0.8, 0.1, 0.05), (0.6, 0.3, 0.05)])
def test_split_ratios_must_sum_to_one(ratios):
    df = CsvSource(SINGLE).read()
    with pytest.raises(PipelineError, match="sum"):
        train_val_test_split(df, ratios[0], ratios[1], ratios[2])


@pytest.mark.parametrize("bad", [(0.0, 0.5, 0.5), (1.0, 0.0, 0.0), (0.5, 0.5, -0.1)])
def test_split_ratio_out_of_range(bad):
    df = CsvSource(SINGLE).read()
    with pytest.raises(PipelineError, match="in \\(0, 1\\)"):
        train_val_test_split(df, bad[0], bad[1], bad[2])


@pytest.mark.parametrize("bad", [("0.7", 0.15, 0.15), (True, 0.5, 0.5)])
def test_split_ratio_wrong_type(bad):
    df = CsvSource(SINGLE).read()
    with pytest.raises(PipelineError):
        train_val_test_split(df, bad[0], bad[1], bad[2])


def test_split_empty_frame():
    df = pd.DataFrame(columns=[TIMESTAMP_COL, SERIES_ID_COL, VALUE_COL])
    with pytest.raises(PipelineError):
        train_val_test_split(df)


# -------------------------------------------------------- normalization


def test_fit_transform_train_stats():
    df = CsvSource(SINGLE).read()
    train, _, _ = train_val_test_split(df, 0.7, 0.15, 0.15)
    normalizer = Normalizer().fit(train)
    scaled = normalizer.transform(train)
    assert scaled[VALUE_COL].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled[VALUE_COL].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_per_series_scaling_is_independent():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 10.0),
            (pd.Timestamp("2024-01-02"), "A", 20.0),
            (pd.Timestamp("2024-01-01"), "B", 100.0),
            (pd.Timestamp("2024-01-02"), "B", 200.0),
        ]
    )
    normalizer = Normalizer().fit(df)
    assert normalizer.scale(np.array([15.0]), "A")[0] == pytest.approx(0.0)
    assert normalizer.scale(np.array([150.0]), "B")[0] == pytest.approx(0.0)
    assert normalizer.scale(np.array([15.0]), "B")[0] != pytest.approx(0.0)


def test_inverse_transform_roundtrip():
    df = CsvSource(MULTI).read()
    train, _, _ = train_val_test_split(df, 0.7, 0.15, 0.15)
    normalizer = Normalizer().fit(train)
    scaled = normalizer.transform(train)
    restored = normalizer.inverse_transform(scaled)
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True),
        train.reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )


def test_unscale_unknown_series_raises():
    df = _frame([(pd.Timestamp("2024-01-01"), "A", 1.0), (pd.Timestamp("2024-01-02"), "A", 2.0)])
    normalizer = Normalizer().fit(df)
    with pytest.raises(PipelineError, match="not present during fit"):
        normalizer.transform(_frame([(pd.Timestamp("2024-01-01"), "B", 1.0)]))


def test_constant_series_scales_to_zero():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 5.0),
            (pd.Timestamp("2024-01-02"), "A", 5.0),
            (pd.Timestamp("2024-01-03"), "A", 5.0),
        ]
    )
    normalizer = Normalizer().fit(df)
    assert normalizer.params["A"] == (5.0, 0.0)
    scaled = normalizer.transform(df)
    assert scaled[VALUE_COL].tolist() == [0.0, 0.0, 0.0]
    restored = normalizer.inverse_transform(scaled)
    assert restored[VALUE_COL].tolist() == [5.0, 5.0, 5.0]


def test_transform_before_fit_raises():
    normalizer = Normalizer()
    with pytest.raises(PipelineError):
        normalizer.scale(np.array([1.0]), "A")


def test_global_mode():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 10.0),
            (pd.Timestamp("2024-01-02"), "A", 20.0),
            (pd.Timestamp("2024-01-01"), "B", 100.0),
            (pd.Timestamp("2024-01-02"), "B", 200.0),
        ]
    )
    normalizer = Normalizer(per_series=False).fit(df)
    mean = (10.0 + 20.0 + 100.0 + 200.0) / 4
    assert normalizer.scale(np.array([mean]), None)[0] == pytest.approx(0.0)
    assert normalizer.params == {"__global__": normalizer.params["__global__"]}
    scaled = normalizer.transform(df)
    assert set(scaled[SERIES_ID_COL]) == {"A", "B"}
    restored = normalizer.inverse_transform(scaled)
    assert restored[VALUE_COL].tolist() == [10.0, 20.0, 100.0, 200.0]


def test_fit_missing_values_raises():
    df = _frame(
        [
            (pd.Timestamp("2024-01-01"), "A", 1.0),
            (pd.Timestamp("2024-01-02"), "A", float("nan")),
        ]
    )
    with pytest.raises(PipelineError, match="missing"):
        Normalizer().fit(df)


def test_fit_non_canonical_raises():
    with pytest.raises(PipelineError):
        Normalizer().fit(pd.DataFrame({"x": [1, 2]}))


def test_scale_unscale_array_roundtrip():
    df = CsvSource(SINGLE).read()
    train, _, _ = train_val_test_split(df, 0.7, 0.15, 0.15)
    normalizer = Normalizer().fit(train)
    original = train[VALUE_COL].to_numpy()
    scaled = normalizer.scale(original, "series_single")
    restored = normalizer.unscale(scaled, "series_single")
    np.testing.assert_allclose(restored, original, atol=1e-9)


def test_transform_input_unchanged():
    df = CsvSource(SINGLE).read()
    train, _, _ = train_val_test_split(df, 0.7, 0.15, 0.15)
    before = train[VALUE_COL].to_numpy().copy()
    normalizer = Normalizer().fit(train)
    normalizer.transform(train)
    np.testing.assert_array_equal(train[VALUE_COL].to_numpy(), before)
