from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.db import (
    count_rows,
    get_meta,
    latest_ts,
    load_artifact,
    observations,
    read_observations,
    save_artifact,
    set_meta,
    upsert_observations,
)


def _frame(values, start="2026-01-01 00:00"):
    idx = pd.date_range(start, periods=len(values), freq="h").as_unit("ns")
    return pd.DataFrame({"pm2_5": values}, index=idx)


def test_upsert_updates_existing_rows_and_does_not_duplicate(engine):
    upsert_observations(engine, "delhi", _frame([1.0, 2.0, 3.0]))
    upsert_observations(engine, "delhi", _frame([9.0, 8.0, 3.0, 4.0]))
    assert count_rows(engine, observations) == 4
    assert read_observations(engine, "delhi")["pm2_5"].tolist() == [9.0, 8.0, 3.0, 4.0]


def test_nan_is_stored_as_null_and_latest_ts_ignores_it(engine):
    frame = _frame([1.0, np.nan, np.nan])
    frame["pm10"] = [5.0, 6.0, 7.0]
    upsert_observations(engine, "delhi", frame)
    df = read_observations(engine, "delhi")
    assert df["pm2_5"].isna().tolist() == [False, True, True]  # NaN round-trips as NULL
    assert latest_ts(engine, "delhi") == pd.Timestamp("2026-01-01 00:00").to_pydatetime()
    assert latest_ts(engine, "delhi", "pm10") == pd.Timestamp("2026-01-01 02:00").to_pydatetime()


def test_rows_with_no_values_at_all_are_not_stored(engine):
    upsert_observations(engine, "delhi", _frame([np.nan, np.nan]))
    assert count_rows(engine, observations) == 0


def test_read_observations_returns_sorted_naive_ns_index(engine):
    upsert_observations(engine, "delhi", _frame([1.0, 2.0, 3.0]))
    df = read_observations(engine, "delhi", start=pd.Timestamp("2026-01-01 01:00").to_pydatetime())
    assert df.index.is_monotonic_increasing and df.index.tz is None
    assert str(df.index.dtype) == "datetime64[ns]" and len(df) == 2


def test_meta_and_artifact_roundtrip(engine):
    set_meta(engine, "k", {"a": 1})
    assert get_meta(engine, "k") == {"a": 1} and get_meta(engine, "missing", 7) == 7
    save_artifact(engine, "m", b"\x00\x01binary", {"v": 2})
    blob, meta, created = load_artifact(engine, "m")
    assert blob == b"\x00\x01binary" and meta == {"v": 2} and created is not None
    save_artifact(engine, "m", b"new", {"v": 3})
    assert load_artifact(engine, "m")[0] == b"new"
